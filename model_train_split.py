# === IMPORT LIBRARY ===
import os
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Input, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



# === KONFIGURASI ===
IMG_SIZE = (150, 150)
CSV_PATH = "processed_dataset_dari_output_baru.csv"
TOTAL_TARGET_DATA = 12000  # Ubah ke jumlah yang lebih pas jika perlu

label_map = {
    0: "A", 1: "Ba", 2: "Ca", 3: "Da", 4: "Ga", 5: "Gha", 6: "Ha", 7: "Ja", 8: "Ka", 9: "La",
    10: "Ma", 11: "Na", 12: "Nga", 13: "Nya", 14: "Pa", 15: "Ra", 16: "Sa", 17: "Ta", 18: "Wa", 19: "Ya"
}

# === LOAD DATASET & SHUFFLE ===
df = pd.read_csv(CSV_PATH).sample(frac=1, random_state=42).reset_index(drop=True)

# Batasi total data agar proporsional (bulat)
if len(df) > TOTAL_TARGET_DATA:
    df = df.iloc[:TOTAL_TARGET_DATA]
print(f"Total data setelah pembatasan: {len(df)}")

# === SPLIT STRATIFIED BALANCE PER KELAS ===
def stratified_balanced_split(df, train_ratio=0.70, test_ratio=0.15, val_ratio=0.15, random_state=42):
    df_train, df_test, df_val = [], [], []
    grouped = df.groupby("label")

    for label, group in grouped:
        group = group.sample(frac=1, random_state=random_state)
        n = len(group)
        n_train = int(n * train_ratio)
        n_test = int(n * test_ratio)
        n_val = n - n_train - n_test

        df_train.append(group.iloc[:n_train])
        df_test.append(group.iloc[n_train:n_train + n_test])
        df_val.append(group.iloc[n_train + n_test:])

    return pd.concat(df_train), pd.concat(df_test), pd.concat(df_val)

df_train, df_test, df_val = stratified_balanced_split(df)

# === LOAD IMAGE DATA ===
def load_images_from_df(df_subset, image_size=(150, 150)):
    images, labels = [], []
    for _, row in df_subset.iterrows():
        img_path = row["file_path"]
        label = row["label"]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Gagal membaca gambar: {img_path}")
            continue
        img = cv2.resize(img, image_size)
        img = img / 255.0
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)

X_train, y_train = load_images_from_df(df_train)
X_test, y_test = load_images_from_df(df_test)
X_val, y_val = load_images_from_df(df_val)

# Reshape untuk CNN
X_train = X_train.reshape(-1, 150, 150, 1)
X_test = X_test.reshape(-1, 150, 150, 1)
X_val = X_val.reshape(-1, 150, 150, 1)

print(f"Train: {len(X_train)}, Test: {len(X_test)}, Val: {len(X_val)}")

# === SIMPAN VALIDASI KE FILE ===
df_val[['file_path', 'label']].to_csv("validation_data1.csv", index=False)
print("Data validasi disimpan ke: validation_data1.csv")

# === SIMPAN TRAINING DAN TESTING KE FILE CSV ===
df_train[['file_path', 'label']].to_csv("training_data1.csv", index=False)
print("Data training disimpan ke: training_data1.csv")

df_test[['file_path', 'label']].to_csv("testing_data1.csv", index=False)
print("Data testing disimpan ke: testing_data1.csv")


# === CNN TRAINING MODEL ===
inputs = Input(shape=(150, 150, 1))
x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(256, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dropout(0.3)(x)
feature_output = x  # yang diambil untuk fitur
output = tf.keras.layers.Dense(20, activation='softmax')(feature_output)

cnn_trainable = Model(inputs=inputs, outputs=output)

cnn_trainable.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

cnn_trainable.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

cnn_model = Model(inputs=cnn_trainable.input, outputs=feature_output)

# === EKSTRAKSI FITUR ===
features_train = cnn_model.predict(X_train)
features_test = cnn_model.predict(X_test)
features_val = cnn_model.predict(X_val)

# === NORMALISASI ===
scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)
features_val = scaler.transform(features_val)

# === TRAINING SVM ===
svm = SVC(kernel='linear', C=1)
svm.fit(features_train, y_train)

# === EVALUASI MODEL ===
y_train_pred = svm.predict(features_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Akurasi data latih: {train_accuracy * 100:.2f}%")

y_val_pred = svm.predict(features_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Akurasi data validasi: {val_accuracy * 100:.2f}%")

y_test_pred = svm.predict(features_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Akurasi data uji: {test_accuracy * 100:.2f}%")

report = classification_report(y_test, y_test_pred, target_names=[label_map[i] for i in sorted(label_map.keys())])
conf_matrix = confusion_matrix(y_test, y_test_pred)

# === SIMPAN LAPORAN ===
train_report = classification_report(y_train, y_train_pred, target_names=[label_map[i] for i in sorted(label_map.keys())])
train_conf_matrix = confusion_matrix(y_train, y_train_pred)

with open("train_report1.txt", "w") as f:
    f.write("=== Laporan Data Latih ===\n\n")
    f.write(f"Akurasi: {train_accuracy * 100:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(train_report)
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(train_conf_matrix))
print("Laporan training disimpan ke: train_report1.txt")

val_report = classification_report(y_val, y_val_pred, target_names=[label_map[i] for i in sorted(label_map.keys())])


with open("val_report1.txt", "w") as f:
    f.write("=== Laporan Data Validasi ===\n\n")
    f.write(f"Akurasi: {val_accuracy * 100:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(val_report)
print("Laporan validasi disimpan ke: val_report1.txt")


test_report = classification_report(y_test, y_test_pred, target_names=[label_map[i] for i in sorted(label_map.keys())])

with open("test_report1.txt", "w") as f:
    f.write("=== Laporan Data Uji ===\n\n")
    f.write(f"Akurasi: {test_accuracy * 100:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(test_report)
print("Laporan testing disimpan ke: test_report1.txt")

with open("evaluation_report1.txt", "w") as f:
    f.write("Laporan Evaluasi Model:\n\n")
    f.write(f"Akurasi data latih: {train_accuracy * 100:.2f}%\n")
    f.write(f"Akurasi data validasi: {val_accuracy * 100:.2f}%\n")
    f.write(f"Akurasi data uji: {test_accuracy * 100:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(conf_matrix))

print("Laporan evaluasi disimpan ke: evaluation_report1.txt")



# === SIMPAN MODEL ===
cnn_model.save("cnn_feature_extractor_trained.keras", save_format='keras')
joblib.dump(scaler, "scaler1.pkl")
joblib.dump(svm, "svm_classifier1.pkl")
print("Model berhasil disimpan.")

# === VISUALISASI CONFUSION MATRIX ===
def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(cm, cmap="Blues")
    plt.colorbar(cax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names)

    for i in range(len(cm)):
        for j in range(len(cm[i])):
            ax.text(j, i, str(cm[i][j]), ha='center', va='center', color='black')

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(conf_matrix, list(label_map.values()))
