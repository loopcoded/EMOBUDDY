from utils.data_preprocessing import load_clean_face_dataset

X_train, y_train, X_val, y_val, X_test, y_test = load_clean_face_dataset("datasets/clean_face_emotions")

print("Shape:", X_train[0].shape)
print("Min:", X_train[0].min(), "Max:", X_train[0].max())
