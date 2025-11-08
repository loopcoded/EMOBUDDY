import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

# Create augmentation generator
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load sample image
img_path = 'datasets/face_emotions/train/happy/Training_1206.jpg'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (48, 48))
img = img.reshape((1,) + img.shape)

# Generate augmented images
plt.figure(figsize=(15, 5))
plt.subplot(1, 6, 1)
plt.imshow(img[0].astype('uint8'))
plt.title('Original')
plt.axis('off')

i = 2
for batch in datagen.flow(img, batch_size=1):
    plt.subplot(1, 6, i)
    plt.imshow(batch[0].astype('uint8'))
    plt.title(f'Augmented {i-1}')
    plt.axis('off')
    i += 1
    if i > 6:
        break

plt.tight_layout()
plt.savefig('augmentation_preview.png')
print("Preview saved: augmentation_preview.png")