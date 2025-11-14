import os
import cv2
import librosa
import hashlib

BASE_DIR = "datasets"

def get_hash(path):
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

def is_image_file(ext):
    return ext in ["jpg", "jpeg", "png"]

def is_audio_file(ext):
    return ext in ["wav", "mp3", "m4a"]

def validate_file(path, is_image):
    try:
        if is_image:
            img = cv2.imread(path)
            return img is not None
        else:
            librosa.load(path, sr=None)
            return True
    except:
        return False

def audit(root_folder, is_image=True):
    print("\n=====================================")
    print(f"   AUDITING DATASET → {root_folder}")
    print("=====================================\n")

    root = os.path.join(BASE_DIR, root_folder)

    class_counts = {}
    corrupted = []
    duplicates = {}
    hash_map = {}

    for split in ["train", "test"]:
        split_path = os.path.join(root, split)

        if not os.path.exists(split_path):
            print(f"❌ Missing folder: {split_path}")
            continue

        print(f"\n🔍 Checking split → {split}")

        for emotion in sorted(os.listdir(split_path)):
            emotion_path = os.path.join(split_path, emotion)

            if not os.path.isdir(emotion_path):
                continue

            files = os.listdir(emotion_path)

            # count
            class_counts[(split, emotion)] = len(files)
            if len(files) == 0:
                print(f"⚠️ EMPTY CLASS: {emotion} in {split}")

            for f in files:
                file_path = os.path.join(emotion_path, f)
                ext = f.split(".")[-1].lower()

                # extension check
                if is_image and not is_image_file(ext):
                    print(f"⚠️ Non-image file: {file_path}")

                if not is_image and not is_audio_file(ext):
                    print(f"⚠️ Non-audio file: {file_path}")

                # corruption check
                if not validate_file(file_path, is_image):
                    corrupted.append(file_path)

                # duplicate check
                h = get_hash(file_path)
                if h:
                    if h not in hash_map:
                        hash_map[h] = file_path
                    else:
                        duplicates.setdefault(hash_map[h], []).append(file_path)

    # ---- REPORT ----
    print("\n------------------------------")
    print("📊 CLASS DISTRIBUTION SUMMARY")
    print("------------------------------")
    for (split, emotion), count in class_counts.items():
        print(f"{split.upper():6} | {emotion:15} | {count}")

    print("\n------------------------------")
    print("📌 DUPLICATE FILE CHECK")
    print("------------------------------")
    if duplicates:
        for orig, dup_list in duplicates.items():
            print(f"Original: {orig}")
            for d in dup_list:
                print(f"  → Duplicate: {d}")
    else:
        print("✔ No duplicates found")

    print("\n------------------------------")
    print("📌 CORRUPTED FILE CHECK")
    print("------------------------------")
    if corrupted:
        print("❌ Corrupted files:")
        for c in corrupted:
            print(c)
    else:
        print("✔ No corrupted files")

    print("\n✔ Audit completed.\n")

# Run for both datasets
audit("face_emotions", is_image=True)
audit("voice_emotions", is_image=False)
