import lmdb
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class LMDBDataset(Dataset):
    """
    PyTorch Dataset for sensor time-series stored in LMDB.

    Expected LMDB entry format:
        {
            "frame": np.ndarray of shape (C, T),
            "label": int,
            "device": str
        }

    Expected metadata under b"__meta__":
        {
            "index_map": [...],
            "devices": {...},
            "sensors": {...},
            "axes": {...},
            "mode": {
                "train": [...],
                "validation": [...],
                "test": [...]
            },
            "norm": {...}   # optional
        }
    """

    def __init__(
        self,
        lmdb_path,
        mode,
        transform=None,
        target_transform=None,
        return_device=False,
        return_key=False,
        as_tensor=True
    ):
        self.lmdb_path = lmdb_path
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.return_device = return_device
        self.return_key = return_key
        self.as_tensor = as_tensor

        self.env = None
        self.meta = self._load_meta()

        if mode not in self.meta["mode"]:
            available = list(self.meta["mode"].keys())
            raise ValueError(f"Mode '{mode}' not found in LMDB metadata. Available modes: {available}")

        self.keys = self.meta["mode"][mode]

        self.devices = self.meta.get("devices", {})
        self.sensors = self.meta.get("sensors", {})
        self.axes = self.meta.get("axes", {})
        self.norm = self.meta.get("norm", {})

    def _open_env(self):
        """
        Open LMDB environment lazily.
        This is safer when DataLoader uses multiple workers.
        """
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )

    def _load_meta(self):
        """
        Load dataset metadata once during initialization.
        """
        env = lmdb.open(
            self.lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        try:
            with env.begin(write=False) as txn:
                meta_bytes = txn.get(b"__meta__")
                if meta_bytes is None:
                    raise RuntimeError("LMDB metadata '__meta__' not found.")
                meta = pickle.loads(meta_bytes)
        finally:
            env.close()
        return meta

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        self._open_env()

        key = self.keys[index]

        with self.env.begin(write=False) as txn:
            data_bytes = txn.get(key)
            if data_bytes is None:
                raise KeyError(f"Key {key} not found in LMDB.")
            sample = pickle.loads(data_bytes)

        frame = sample["frame"]
        label = sample["label"]
        device = sample.get("device", None)

        if self.transform is not None:
            frame = self.transform(frame)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.as_tensor:
            frame = torch.as_tensor(frame, dtype=torch.float32)
            label = torch.as_tensor(label, dtype=torch.long)

        outputs = [frame, label]

        if self.return_device:
            outputs.append(device)

        if self.return_key:
            outputs.append(key)

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def __getstate__(self):
        """
        Prevent LMDB env from being pickled when DataLoader spawns workers.
        """
        state = self.__dict__.copy()
        state["env"] = None
        return state

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
