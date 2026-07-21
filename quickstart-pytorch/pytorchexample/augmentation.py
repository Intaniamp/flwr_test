import cv2
import os
import albumentations as A
from glob import glob

# Setup direktori sesuai dengan komputer kamu
input_folder = r"C:\coolyeah\flwr-test\real_dataset\normal"
output_folder = r"C:\coolyeah\flwr-test\real_dataset\normal_augmented"

# Buat folder output jika belum ada
os.makedirs(output_folder, exist_ok=True)

# Definisi kombinasi augmentasi secara acak
transform = A.Compose([
    A.HorizontalFlip(p=0.5), # Peluang 50% gambar dibalik secara horizontal
    A.VerticalFlip(p=0.2), # Peluang 20% dibalik vertikal
    A.RandomBrightnessContrast(p=0.3), # Ubah kecerahan/kontras
    A.Rotate(limit=45, p=0.5), # Putar gambar maksimal 45 derajat
    A.GaussNoise(p=0.2), # Tambah sedikit noise/bintik agar model lebih kebal
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5)
])

# Ambil semua file berformat jpg, jpeg, atau png (mengabaikan huruf besar/kecil)
image_extensions = ('*.jpeg', '*.jpg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
images = []
for ext in image_extensions:
    images.extend(glob(os.path.join(input_folder, ext)))

target_jumlah = 5 # Berapa banyak variasi baru yang mau dibuat dari 1 gambar asli

for img_path in images:
    # Ambil nama file aslinya (misal: "blast (1)")
    filename = os.path.basename(img_path).rsplit('.', 1)[0]
    ext = os.path.basename(img_path).rsplit('.', 1)[1]
    
    # Baca gambar
    image_bgr = cv2.imread(img_path)
    
    # Safety check: Lewati kalau gambar gagal dibaca
    if image_bgr is None:
        print(f"⚠️ Gagal membaca gambar: {img_path}. Melewati file ini...")
        continue
        
    # Konversi ke RGB untuk Albumentations
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    print(f"Memproses augmentasi untuk: {filename}.{ext}")

    # (OPSIONAL TAPI PENTING) Simpan gambar aslinya ke folder output juga
    # Biar folder output berisi data asli + data augmentasi
    cv2.imwrite(os.path.join(output_folder, f"{filename}_original.jpg"), image_bgr)

    for i in range(target_jumlah):
        # Terapkan augmentasi
        transformed = transform(image=image_rgb)
        transformed_image_bgr = cv2.cvtColor(transformed["image"], cv2.COLOR_RGB2BGR)
        
        # Simpan gambar baru ke folder output
        new_filename = f"{filename}_aug_{i}.jpg"
        cv2.imwrite(os.path.join(output_folder, new_filename), transformed_image_bgr)

print("✅ Proses augmentasi selesai! Cek folder normal_augmented.")