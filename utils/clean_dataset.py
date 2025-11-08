import os
import cv2
from pathlib import Path

def clean_face_dataset(dataset_path):
    """Remove corrupted or invalid images."""
    print("Cleaning face dataset...")
    
    removed = 0
    total = 0
    
    for emotion_folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, emotion_folder)
        
        if not os.path.isdir(folder_path):
            continue
        
        for img_file in os.listdir(folder_path):
            total += 1
            img_path = os.path.join(folder_path, img_file)
            
            try:
                img = cv2.imread(img_path)
                if img is None or img.size == 0:
                    os.remove(img_path)
                    removed += 1
                    print(f"Removed: {img_path}")
            except Exception as e:
                os.remove(img_path)
                removed += 1
                print(f"Removed (error): {img_path}")
    
    print(f"\nCleaning complete:")
    print(f"  Total images: {total}")
    print(f"  Removed: {removed}")
    print(f"  Valid: {total - removed}")

# Run cleaning
clean_face_dataset('datasets/face_emotions/train')
clean_face_dataset('datasets/face_emotions/test')