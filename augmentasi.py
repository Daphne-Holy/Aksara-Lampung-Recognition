import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path ke folder gambar asli
source_folder = 'C:/Users/Daphne Holy/Desktop/Projek Skripsi/Gambar Aksara Lampung/Ya'
destination_folder = 'C:/Users/Daphne Holy/Desktop/Projek Skripsi/Augmentasi/Ya'

# Pastikan folder tujuan ada
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Augmentasi menggunakan ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=15,      # Rotasi gambar maksimal 30 derajat
    width_shift_range=0.2,  # Pergeseran lebar hingga 20%
    height_shift_range=0.2, # Pergeseran tinggi hingga 20%
    fill_mode='nearest'     # Isi piksel kosong
)

# Fungsi augmentasi dan penyimpanan
def augment_images(source_folder, destination_folder, total_target=600):
    images = os.listdir(source_folder)
    total_images = len(images)
    augment_per_image = -(-total_target // total_images)  # Pembulatan ke atas

    for image_file in images:
        img_path = os.path.join(source_folder, image_file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (150, 150))  # Resize semua gambar ke 150x150
        img = np.expand_dims(img, axis=0)  # Tambahkan dimensi batch
        
        # Augmentasi gambar
        count = 0
        for batch in datagen.flow(img, batch_size=1, save_to_dir=destination_folder,
                                  save_prefix='aug', save_format='jpg'):
            count += 1
            if count >= augment_per_image:  # Batasi jumlah augmentasi per gambar
                break

augment_images(source_folder, destination_folder, total_target=600)
print("Augmentasi selesai. Semua gambar disimpan di folder tujuan!")

    # shear_range=0.2,        # Distorsi gambar
    # zoom_range=0.1,         # Zoom in/out gambar
    # horizontal_flip=True,   # Membalik gambar secara horizontal