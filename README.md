## Deskripsi Proyek
Project ini merupakan implementasi Deep Learning untuk klasifikasi citra menggunakan arsitektur **InceptionV3** dengan pendekatan **Transfer Learning**. Model digunakan untuk mengklasifikasikan gambar pemandangan ke dalam beberapa kategori menggunakan dataset **Intel Image Classification** dari Kaggle.

Arsitektur InceptionV3 dipilih karena mampu mengekstraksi fitur dari berbagai skala menggunakan kombinasi filter konvolusi berukuran **1×1, 3×3, dan 5×5** dalam satu modul, sehingga meningkatkan performa klasifikasi citra.

Project ini dibuat sebagai bagian dari praktikum **Deep Learning** di **Fakultas Vokasi Universitas Airlangga**.

---

## Tujuan Proyek

Tujuan dari project ini adalah:

- Mengimplementasikan Deep Learning dengan arsitektur **InceptionV3** untuk klasifikasi citra.
- Menerapkan teknik **Transfer Learning** agar proses pelatihan lebih efisien.
- Menggunakan **Data Augmentation** untuk meningkatkan kemampuan generalisasi model.
- Mengevaluasi performa model menggunakan berbagai metrik evaluasi.

---

## Dataset

Dataset yang digunakan adalah **Intel Image Classification Dataset** dari Kaggle.

Dataset ini berisi gambar pemandangan yang terbagi dalam **6 kelas**, yaitu:

- buildings  
- forest  
- glacier  
- mountain  
- sea  
- street  
---

## Arsitektur Model

Model yang digunakan adalah **InceptionV3** dengan pendekatan **Transfer Learning**.

Tahapan utama:

1. Menggunakan **pre-trained model InceptionV3** dari dataset ImageNet.
2. Menghapus layer output asli dari model.
3. Menambahkan **Fully Connected Layer baru** sesuai jumlah kelas dataset.
4. Melakukan **fine-tuning pada beberapa layer terakhir** untuk menyesuaikan dengan dataset.

---

## Hyperparameter Training

| Parameter | Nilai |
|----------|------|
| Batch Size | 32 |
| Epoch | 15 |
| Learning Rate | 1e-4 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Weight Decay | 1e-4 |

---

## Data Augmentation

Untuk meningkatkan variasi data training digunakan beberapa teknik augmentasi:

- Random Horizontal Flip
- Random Affine (rotation, scale, shear)
- Resize (299 × 299)
- Normalize

Data **validation dan test tidak menggunakan augmentasi** agar hasil evaluasi tetap konsisten.

---

## Hasil Model

Hasil training menunjukkan bahwa model mampu mencapai performa yang cukup baik.

Beberapa metrik evaluasi yang diperoleh:

- Accuracy : sekitar **87%**
- Precision : sekitar **0.87**
- Recall : sekitar **0.87**
- F1-score : sekitar **0.87**

Model memiliki performa terbaik pada kelas **forest** dan **street**, sedangkan kesalahan prediksi lebih sering terjadi pada kelas **mountain** dan **glacier** karena memiliki karakteristik visual yang mirip.

---

## Visualisasi Training

Selama proses training:

- **Accuracy meningkat secara stabil**
- **Loss menurun secara konsisten**

Hal ini menunjukkan bahwa model berhasil belajar dengan baik tanpa terjadi overfitting yang signifikan.

---

## Kesimpulan

Model **InceptionV3 dengan Transfer Learning** berhasil diterapkan untuk klasifikasi citra pada dataset Intel Image Classification. Dengan memanfaatkan augmentasi data dan model pra-latih, sistem mampu mencapai performa yang baik dengan akurasi tinggi.

Pendekatan ini dapat digunakan sebagai dasar untuk pengembangan sistem klasifikasi citra lainnya di bidang computer vision.
