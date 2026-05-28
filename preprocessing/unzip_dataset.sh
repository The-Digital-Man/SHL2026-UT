# Define the target SHL 2026 zip files
# These should be downloaded from -> http://www.shl-dataset.org/challenge-2026/
ZIP_FILES=(
    "SHL-2026-Train_Bag.zip"
    "SHL-2026-Train_Hand.zip"
    "SHL-2026-Train_Hips.zip"
    "SHL-2026-Train_Torso.zip"
    "SHL-2026-Validation.zip"
    "SHL-2026-Test.zip"
)

echo "=== Starting Quiet Extraction Pipeline for SHL 2026 ==="

for archive in "${ZIP_FILES[@]}"; do
    if [[ -f "$archive" ]]; then
        echo "Extracting: $archive..."
        # -q  : Quiet mode (suppresses standard output)
        # -o  : Overwrite existing files without prompting (ensures idempotency)
        unzip -qo "$archive"

        # Check exit status of the previous command ($?)
        if [[ $? -eq 0 ]]; then
            echo "Successfully extracted: $archive"
        else
            echo "ERROR: Extraction failed for $archive" >&2
        fi
    else
        echo "WARNING: $archive not found in the current directory. Skipping." >&2
    fi
done

echo "=== Extraction Pipeline Finished ==="
