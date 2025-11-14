import os
import shutil
import hashlib
import random
from collections import defaultdict

# -----------------------------
# CONFIG
# -----------------------------
DATASET_ROOT = "datasets/face_emotions"
OUTPUT_ROOT  = "datasets/clean_face_emotions"

SPLIT_RATIOS = {
    "train": 0.80,
    "val":   0.10,
    "test":  0.10
}

random.seed(42)


# -----------------------------
# Utility functions
# -----------------------------
def compute_hash(path):
    """Compute MD5 hash for duplicate detection."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# -----------------------------
# Step 1: Collect all images
# -----------------------------
print("STEP 1 — Collecting files...")

all_files = defaultdict(list)

for split in ["train", "test"]:
    split_path = os.path.join(DATASET_ROOT, split)
    for emotion in os.listdir(split_path):
        emotion_path = os.path.join(split_path, emotion)
        if not os.path.isdir(emotion_path):
            continue
        for file in os.listdir(emotion_path):
            full_path = os.path.join(emotion_path, file)
            if os.path.isfile(full_path):
                all_files[file].append((emotion, full_path))

print(f"Total unique filenames found: {len(all_files)}")


# -----------------------------
# Step 2: Deduplicate by content hash
# -----------------------------
print("STEP 2 — Deduplicating by content...")

hash_map = defaultdict(list)
label_map = defaultdict(list)

for file_entries in all_files.values():
    for emotion, path in file_entries:
        h = compute_hash(path)
        hash_map[h].append((emotion, path))

print(f"Unique image hashes: {len(hash_map)}")


# -----------------------------
# Step 3: Resolve label conflicts
# -----------------------------
print("STEP 3 — Resolving label conflicts...")

clean_dataset = []  # (emotion, path-to-copy)

for h, entries in hash_map.items():
    emotions = [e for e, p in entries]
    # Choose the most common label
    correct_label = max(set(emotions), key=emotions.count)
    # Keep only ONE copy (first file)
    selected_path = entries[0][1]
    clean_dataset.append((correct_label, selected_path))

print(f"Final cleaned images: {len(clean_dataset)}")


# -----------------------------
# Step 4: Group by emotion
# -----------------------------
emotion_groups = defaultdict(list)
for emotion, file_path in clean_dataset:
    emotion_groups[emotion].append(file_path)


# -----------------------------
# Step 5: Create output structure
# -----------------------------
print("STEP 5 — Creating cleaned directory...")

if os.path.exists(OUTPUT_ROOT):
    shutil.rmtree(OUTPUT_ROOT)

for split in SPLIT_RATIOS:
    for emotion in emotion_groups:
        ensure_dir(os.path.join(OUTPUT_ROOT, split, emotion))


# -----------------------------
# Step 6: Split into train/val/test
# -----------------------------
print("STEP 6 — Splitting dataset...")

for emotion, files in emotion_groups.items():
    random.shuffle(files)

    total = len(files)
    n_train = int(total * SPLIT_RATIOS["train"])
    n_val   = int(total * SPLIT_RATIOS["val"])
    # rest goes to test
    n_test  = total - n_train - n_val

    train_files = files[:n_train]
    val_files   = files[n_train : n_train+n_val]
    test_files  = files[n_train+n_val :]

    # copy files
    for f in train_files:
        shutil.copy(f, os.path.join(OUTPUT_ROOT, "train", emotion))

    for f in val_files:
        shutil.copy(f, os.path.join(OUTPUT_ROOT, "val", emotion))

    for f in test_files:
        shutil.copy(f, os.path.join(OUTPUT_ROOT, "test", emotion))

    print(f"{emotion}: {total} → train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

print("\n🎉 CLEANUP COMPLETE")
print(f"Clean dataset available at: {OUTPUT_ROOT}")
