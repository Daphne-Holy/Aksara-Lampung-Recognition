import os
import cv2
import numpy as np
import pandas as pd

def preprocess_image(image_path, image_size=(150, 150)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Gagal memuat gambar: {image_path}")
        return None
    
    # Resize gambar
    image = cv2.resize(image, image_size)
    
    # Konversi ke grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Denoising dengan Gaussian Blur
    denoised_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
    
    # Bersihkan latar belakang dengan segmentasi sederhana (thresholding)
    _, binary_mask = cv2.threshold(denoised_image, 30, 255, cv2.THRESH_BINARY)
    cleaned_image = cv2.bitwise_and(denoised_image, binary_mask)
    
    # Sharpening menggunakan filter kernel
    sharpening_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
    sharpened_image = cv2.filter2D(cleaned_image, -1, sharpening_kernel)
    
    # Normalisasi ke rentang [0, 1]
    normalized_image = sharpened_image / 255.0
    
    return normalized_image

def load_and_preprocess_images(folder_path, output_folder=None, image_size=(150, 150)):
    images = []
    labels = []
    file_paths = []  # Inisialisasi file_paths
    class_names = []

    if not os.path.exists(folder_path):
        print(f"Folder input tidak ditemukan: {folder_path}")
        return None, None, None, None

    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for label, folder in enumerate(os.listdir(folder_path)):
        folder_full_path = os.path.join(folder_path, folder)
        if os.path.isdir(folder_full_path):
            class_names.append(folder)
            # Buat folder output untuk setiap kelas
            output_class_folder = os.path.join(output_folder, folder)
            if not os.path.exists(output_class_folder):
                os.makedirs(output_class_folder)
            for file_name in os.listdir(folder_full_path):
                file_path = os.path.join(folder_full_path, file_name)
                if file_path.endswith(('.jpg', '.png')):
                    preprocessed_image = preprocess_image(file_path, image_size)
                    if preprocessed_image is not None:
                        images.append(preprocessed_image)
                        labels.append(label)
                        file_paths.append(os.path.join(output_class_folder, file_name))  # Simpan file path di folder output
                        # Simpan gambar hasil preprocessing ke folder kelas masing-masing
                        output_file_path = os.path.join(output_class_folder, file_name)
                        cv2.imwrite(output_file_path, (preprocessed_image * 255).astype(np.uint8))

    return np.array(images), np.array(labels), class_names, file_paths

def generate_csv_from_output(folder_path, csv_output):
    file_paths = []
    labels = []
    class_names = []

    if not os.path.exists(folder_path):
        print(f"Folder output tidak ditemukan: {folder_path}")
        return None

    for label, folder in enumerate(os.listdir(folder_path)):
        folder_full_path = os.path.join(folder_path, folder)
        if os.path.isdir(folder_full_path):
            class_names.append(folder)
            for file_name in os.listdir(folder_full_path):
                file_path = os.path.join(folder_full_path, file_name)
                if file_path.endswith(('.jpg', '.png')):
                    file_paths.append(file_path)
                    labels.append(label)

    # Simpan file_paths dan label ke CSV
    if file_paths and labels:
        df = pd.DataFrame({"file_path": file_paths, "label": labels})
        df.to_csv(csv_output, index=False)
        print(f"File path dan label berhasil disimpan ke: {csv_output}")
    else:
        print("Tidak ada file untuk disimpan ke CSV!")

# Path input, output, dan CSV
input_folder = "C:/Users/Daphne Holy/Desktop/Projek Skripsi/Augmentasi/rev_augment"  # CSV awal diambil dari sini
output_folder = "C:/Users/Daphne Holy/Desktop/Projek Skripsi/Barubgt"  # Folder hasil preprocessing
csv_output = "processed_dataset_dari_output_baru.csv"

# Proses gambar
images, labels, class_names, file_paths = load_and_preprocess_images(input_folder, output_folder)

# Buat CSV dari folder output
generate_csv_from_output(output_folder, csv_output)

if images is not None:
    print(f"Total gambar yang diproses: {len(images)}")
    print(f"Labels: {class_names}")
else:
    print("Tidak ada gambar yang berhasil diproses!")
