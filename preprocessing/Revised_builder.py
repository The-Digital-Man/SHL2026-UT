import os
import lmdb
import pickle
import numpy as np

from tqdm import tqdm
from scipy.signal import butter, filtfilt


# ============================================================
# CONFIG
# ============================================================

FS = 100

SENSORS = [
    "Acc_x.txt",
    "Acc_y.txt",
    "Acc_z.txt",
    "Gyr_x.txt",
    "Gyr_y.txt",
    "Gyr_z.txt",
    "Mag_x.txt",
    "Mag_y.txt",
    "Mag_z.txt",
]


# ============================================================
# FILTERING
# ============================================================

def bandpass_filter_1d(x, fs=100, low=0.1, high=20.0, order=4):
    """
    Bandpass filter preserving:
      - gravity / slow motion (>0.1 Hz)
      - human motion (<20 Hz)

    Based on PSD analysis:
      - strong low-frequency dominance
      - little useful information above 20Hz
    """

    nyq = 0.5 * fs

    low = max(low, 1e-5)
    high = min(high, nyq - 1e-5)

    b, a = butter(
        order,
        [low / nyq, high / nyq],
        btype="band"
    )

    return filtfilt(b, a, x)


def filter_sample_keep_100hz(
    frame,
    fs=100,
    low=0.1,
    high=20.0,
    order=4
):
    """
    frame shape: (C, T)

    Returns:
        same shape, filtered
    """

    out = np.empty_like(frame, dtype=np.float32)

    for ch in range(frame.shape[0]):
        out[ch] = bandpass_filter_1d(
            frame[ch],
            fs=fs,
            low=low,
            high=high,
            order=order
        )

    return out.astype(np.float32)


# ============================================================
# LMDB BUILDER
# ============================================================

class LMDBBuilder:
    """
    SHL2026 LMDB builder.

    Final preprocessing strategy based on diagnostics
    
    """

    def __init__(
        self,
        root_dir,
        lmdb_path,
        map_size=1e12,

        fs=100,

        preprocessing=True,

        bp_low=0.1,
        bp_high=20.0,
        bp_order=4,

        clip_percentile=99.5,

        eps=1e-8,
    ):

        self.root_dir = root_dir
        self.lmdb_path = lmdb_path

        self.env = lmdb.open(
            lmdb_path,
            map_size=int(map_size)
        )

        self.fs = fs

        self.preprocessing = preprocessing

        self.bp_low = bp_low
        self.bp_high = bp_high
        self.bp_order = bp_order

        self.clip_percentile = clip_percentile

        self.eps = eps

        self.index = 0

        self.index_encoder = lambda idx: f"{idx:09d}".encode()

        self.meta = {
            "index_map": [],
            "devices": {},
            "sensors": {},
            "axes": {},
            "mode": {},
            "norm": {},
        }

    # ========================================================
    # CLEANUP
    # ========================================================

    def close(self):
        self.env.close()

    # ========================================================
    # FILE HELPERS
    # ========================================================

    def _parse_filename(self, filename):

        name = filename[:-4] if filename.endswith(".txt") else filename

        parts = name.split("_")

        if len(parts) < 2:
            raise ValueError(
                f"Unexpected sensor filename: {filename}"
            )

        return parts[0], parts[1]

    def _collect_device_files(self, device_path):

        filenames = sorted([
            f for f in os.listdir(device_path)
            if f != "Label.txt" and f.endswith(".txt")
        ])

        data_paths = [
            os.path.join(device_path, f)
            for f in filenames
        ]

        label_path = os.path.join(device_path, "Label.txt")

        for f in filenames:

            sensor, axis = self._parse_filename(f)

            if sensor not in self.meta["sensors"]:
                self.meta["sensors"][sensor] = len(self.meta["sensors"])

            if axis not in self.meta["axes"]:
                self.meta["axes"][axis] = len(self.meta["axes"])

        return data_paths, label_path

    # ========================================================
    # FRAME ITERATOR
    # ========================================================

    def _iter_frames_from_device(
        self,
        data_paths,
        label_path
    ):

        data_files = [open(p, "r") for p in data_paths]

        label_file = (
            open(label_path, "r")
            if os.path.exists(label_path)
            else None
        )

        try:

            while True:

                lines = [f.readline() for f in data_files]

                # EOF
                if not lines[0]:
                    break

                # misalignment safety
                if any(l == "" for l in lines):
                    break

                # --------------------------------------------
                # Parse vectors
                # --------------------------------------------

                vectors = []

                valid = True

                for line in lines:

                    try:
                        # FIX: test is comma sep
                        clean_line = line.replace(',', ' ')
                        vec = np.array(
                            clean_line.strip().split(),
                            dtype=np.float32
                        )

                    except Exception:
                        valid = False
                        break

                    if len(vec) == 0:
                        valid = False
                        break

                    vectors.append(vec)

                if not valid:
                    continue

                lengths = [len(v) for v in vectors]

                # ensure all channels same length
                if len(set(lengths)) != 1:
                    continue

                frame = np.stack(vectors, axis=0)

                # expected shape
                if frame.shape[0] != 9:
                    continue

                # expected window size
                if frame.shape[1] != 500:
                    continue

                # --------------------------------------------
                # Labels
                # --------------------------------------------

                if label_file is not None:

                    label_line = label_file.readline()

                    arr = np.fromstring(
                        label_line.strip(),
                        sep=" ",
                        dtype=np.float32
                    )

                    label = int(arr[0] - 1) if len(arr) else -1

                else:
                    label = -1

                # --------------------------------------------
                # NaN / Inf Handling
                # --------------------------------------------

                frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)

                # --------------------------------------------
                # Filtering
                # --------------------------------------------

                if self.preprocessing:

                    frame = filter_sample_keep_100hz(
                        frame,
                        fs=self.fs,
                        low=self.bp_low,
                        high=self.bp_high,
                        order=self.bp_order
                    )

                yield frame.astype(np.float32), label

        finally:

            for f in data_files:
                f.close()

            if label_file is not None:
                label_file.close()

    # ========================================================
    # GLOBAL STATISTICS
    # ========================================================

    def compute_normalization_stats(
        self,
        modes=("train", "validation")
    ):
        """
        Compute:

            mean
            std
            percentile clip thresholds

        per channel.
        """

        print("\nComputing normalization statistics...")

        sums = None
        sq_sums = None
        counts = None

        # For robust clipping
        channel_values = [[] for _ in range(9)]

        for _, _, frame, _ in tqdm(
            self._iter_all_frames(modes),
            desc="Statistics pass"
        ):

            if sums is None:

                c = frame.shape[0]

                sums = np.zeros(c, dtype=np.float64)
                sq_sums = np.zeros(c, dtype=np.float64)
                counts = np.zeros(c, dtype=np.int64)

            sums += frame.sum(axis=1)

            sq_sums += (frame ** 2).sum(axis=1)

            counts += frame.shape[1]

            # subsample for percentile estimation
            for ch in range(9):

                vals = frame[ch]

                if len(channel_values[ch]) < 2_000_000:
                    channel_values[ch].extend(vals[::10])

        means = sums / counts

        variances = sq_sums / counts - means ** 2
        stds = np.sqrt(np.maximum(variances, self.eps))

        # robust clipping thresholds
        clip_values = []

        for ch in range(9):

            vals = np.asarray(channel_values[ch])

            clip_val = np.percentile(
                np.abs(vals),
                self.clip_percentile
            )

            clip_values.append(clip_val)

        clip_values = np.asarray(
            clip_values,
            dtype=np.float32
        )

        return (
            means.astype(np.float32),
            stds.astype(np.float32),
            clip_values.astype(np.float32)
        )

    # ========================================================
    # ITERATE ALL FRAMES
    # ========================================================

    def _iter_all_frames(self, modes):

        for mode in modes:

            mode_path = os.path.join(
                self.root_dir,
                mode
            )

            if not os.path.isdir(mode_path):
                continue

            for device in sorted(os.listdir(mode_path)):

                device_path = os.path.join(
                    mode_path,
                    device
                )

                if not os.path.isdir(device_path):
                    continue

                data_paths, label_path = (
                    self._collect_device_files(device_path)
                )

                for frame, label in self._iter_frames_from_device(
                    data_paths,
                    label_path
                ):

                    yield mode, device, frame, label

    # ========================================================
    # NORMALIZATION
    # ========================================================

    def normalize_frame(
        self,
        frame,
        means,
        stds,
        clip_values
    ):
        """
        Robust preprocessing pipeline:

        1. percentile clipping
        2. z-score normalization
        """

        frame = frame.copy()

        # --------------------------------------------
        # robust clipping
        # --------------------------------------------

        for ch in range(frame.shape[0]):

            clip_val = clip_values[ch]

            frame[ch] = np.clip(
                frame[ch],
                -clip_val,
                clip_val
            )

        # --------------------------------------------
        # z-score normalization
        # --------------------------------------------

        frame = (
            frame - means[:, None]
        ) / (
            stds[:, None] + self.eps
        )

        return frame.astype(np.float32)

    # ========================================================
    # STORE ENTRY
    # ========================================================

    def _store_entry(self, txn, entry):

        key = self.index_encoder(self.index)

        txn.put(
            key,
            pickle.dumps(entry)
        )

        self.meta["index_map"].append(key)

        self.index += 1

    # ========================================================
    # BUILD
    # ========================================================

    def build(
        self,
        modes=("train", "validation", "test"),
        normalize=True
    ):

        # ----------------------------------------------------
        # Compute normalization stats
        # ----------------------------------------------------

        if normalize:

            means, stds, clip_values = (
                self.compute_normalization_stats(
                    modes=("train", "validation")
                )
            )

            self.meta["norm"] = {
                "type": "robust_zscore",

                "mean": means,
                "std": stds,

                "clip_percentile": self.clip_percentile,
                "clip_values": clip_values,
            }

            print("\nNormalization statistics:")
            print("Means:", means)
            print("Stds :", stds)
            print("Clip :", clip_values)

        else:

            means = None
            stds = None
            clip_values = None

            self.meta["norm"] = {
                "type": "none"
            }

        # ----------------------------------------------------
        # Write LMDB
        # ----------------------------------------------------

        with self.env.begin(write=True) as txn:

            for mode in modes:

                mode_path = os.path.join(
                    self.root_dir,
                    mode
                )

                if not os.path.isdir(mode_path):
                    continue

                start_index = self.index

                subdirs = [
                    d for d in os.listdir(mode_path)
                    if os.path.isdir(
                        os.path.join(mode_path, d)
                    )
                ]

                # =================================================
                # TRAIN / VALIDATION
                # =================================================

                if len(subdirs) > 0:

                    for device_num, device in enumerate(sorted(subdirs)):

                        device_path = os.path.join(
                            mode_path,
                            device
                        )

                        self.meta["devices"][device] = device_num

                        data_paths, label_path = (
                            self._collect_device_files(device_path)
                        )

                        for frame, label in tqdm(
                            self._iter_frames_from_device(
                                data_paths,
                                label_path
                            ),
                            desc=f"Write {mode}/{device}"
                        ):

                            if normalize:

                                frame = self.normalize_frame(
                                    frame,
                                    means,
                                    stds,
                                    clip_values
                                )

                            entry = {
                                "frame": frame.astype(np.float32),
                                "label": int(label),
                                "device": device,
                            }

                            self._store_entry(txn, entry)

                # =================================================
                # TEST
                # =================================================

                else:

                    device = "Unknown"

                    if device not in self.meta["devices"]:
                        self.meta["devices"][device] = -1

                    data_paths, label_path = (
                        self._collect_device_files(mode_path)
                    )

                    for frame, label in tqdm(
                        self._iter_frames_from_device(
                            data_paths,
                            label_path
                        ),
                        desc=f"Write {mode}/Flat"
                    ):

                        if normalize:

                            frame = self.normalize_frame(
                                frame,
                                means,
                                stds,
                                clip_values
                            )

                        entry = {
                            "frame": frame.astype(np.float32),
                            "label": int(label),
                            "device": device,
                        }

                        self._store_entry(txn, entry)

                self.meta["mode"][mode] = [
                    self.index_encoder(i)
                    for i in range(start_index, self.index)
                ]

            # --------------------------------------------
            # store metadata
            # --------------------------------------------

            txn.put(
                b"__meta__",
                pickle.dumps(self.meta)
            )

        print("\nLMDB build complete.")
        print(f"Total samples: {self.index}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    current_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    builder = LMDBBuilder(

        root_dir=current_dir,

        lmdb_path=os.path.join(
            current_dir,
            "shl2026_dataset.lmdb"
        ),

        preprocessing=False,

        # based on PSD analysis
        bp_low=0.1,
        bp_high=20.0,

        # robust normalization
        clip_percentile=99.5,
    )

    builder.build(
        modes=("train", "validation", "test"),
        normalize=True
    )

    builder.close()
