import os
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataset(dataset_path, emotions):
    """Analyze dataset distribution."""
    
    stats = {emotion: 0 for emotion in emotions}
    
    for emotion in emotions:
        emotion_path = os.path.join(dataset_path, emotion)
        if os.path.exists(emotion_path):
            stats[emotion] = len([f for f in os.listdir(emotion_path) 
                                 if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    # Plot distribution
    plt.figure(figsize=(12, 6))
    plt.bar(stats.keys(), stats.values(), color='skyblue')
    plt.xlabel('Emotion')
    plt.ylabel('Number of Samples')
    plt.title('Dataset Distribution')
    plt.xticks(rotation=45)
    
    for i, (emotion, count) in enumerate(stats.items()):
        plt.text(i, count, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('dataset_distribution.png')
    
    print("\nDataset Statistics:")
    print("-" * 40)
    total = sum(stats.values())
    for emotion, count in stats.items():
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{emotion:15s}: {count:4d} ({percentage:5.1f}%)")
    print("-" * 40)
    print(f"{'Total':15s}: {total:4d}")
    
    # Check balance
    if total > 0:
        percentages = [count/total*100 for count in stats.values()]
        if max(percentages) - min(percentages) > 30:
            print("\n⚠️ WARNING: Dataset is imbalanced!")
            print("   Consider collecting more data for underrepresented emotions.")
    
    return stats

# Analyze both datasets
emotions = ['angry', 'happy', 'disgust', 'fear', 'sad', 'surprise', 'neutral']
print("TRAINING SET:")
analyze_dataset('datasets/face_emotions/train', emotions)
print("\n\nTEST SET:")
analyze_dataset('datasets/face_emotions/test', emotions)