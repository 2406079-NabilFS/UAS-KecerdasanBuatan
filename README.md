# TB AI Flowers - Klasifikasi Jenis Bunga dengan Deep Learning

---

## 1. Deskripsi Projek

Proyek **TB AI Flowers** adalah implementasi sistem klasifikasi gambar bunga menggunakan **Convolutional Neural Network (CNN)** dengan arsitektur **MobileNetV2** berbasis framework **TensorFlow/Keras**. Sistem ini mampu mengenali empat jenis bunga secara otomatis melalui gambar yang diunggah pengguna.

Proyek ini merupakan Tugas Besar mata kuliah Kecerdasan Buatan, Program Studi Teknik Informatika, Tahun Akademik 2024/2025.

Sebagai perbandingan, proyek ini juga dianalisis bersama proyek **TB AIIII Flowers** yang menggunakan arsitektur **Vision Transformer (ViT-Base/16)** berbasis **PyTorch** untuk melihat perbedaan performa antara pendekatan CNN dan Transformer dalam klasifikasi gambar berskala kecil.

---

## 2. Daftar Isi

1. Deskripsi Projek
2. Daftar Isi
3. Latar Belakang
4. Tujuan Projek
5. Dataset
6. Model yang Digunakan
7. Hasil Pengujian
8. Perbandingan Model
9. Referensi

---

## 3. Latar Belakang

Perkembangan teknologi kecerdasan buatan dalam bidang **computer vision** telah mengalami kemajuan pesat. Salah satu aplikasi signifikannya adalah klasifikasi gambar, yang memungkinkan komputer mengenali objek visual secara otomatis. Di Indonesia, keanekaragaman hayati yang melimpah menciptakan kebutuhan akan sistem identifikasi otomatis untuk membantu berbagai pihak mengenali jenis-jenis bunga.

**Deep learning**, khususnya **Convolutional Neural Network (CNN)** , telah menjadi standar sejak keberhasilan AlexNet (Krizhevsky et al., 2012). Arsitektur **MobileNetV2** (Sandler et al., 2018) menawarkan efisiensi komputasi tinggi dengan residual inverted bottleneck, cocok untuk perangkat dengan sumber daya terbatas. Sementara itu, **Vision Transformer (ViT)** (Dosovitskiy et al., 2021) menunjukkan bahwa arsitektur transformer murni dapat mencapai performa state-of-the-art dengan mekanisme self-attention pada patch gambar.

Penelitian ini menganalisis perbandingan kedua pendekatan dalam klasifikasi empat jenis bunga: **Bunga Matahari, Bunga Mawar, Bunga Melati, dan Bunga Tulip**.

---

## 4. Tujuan Projek

1. Membangun model deep learning yang mampu mengklasifikasikan gambar bunga ke dalam empat kategori dengan akurasi tinggi.
2. Membandingkan performa dan efisiensi antara arsitektur CNN (MobileNetV2) dengan arsitektur Transformer (ViT-Base) pada dataset berskala kecil (±400 gambar).
3. Mengimplementasikan sistem berbasis web menggunakan **Streamlit** untuk klasifikasi bunga secara interaktif.
4. Menganalisis trade-off antara **akurasi, kecepatan inferensi, dan kebutuhan komputasi** masing-masing arsitektur.

---

## 5. Dataset

### 5.1 Sumber Data

Dataset bersumber dari platform **Kaggle** ("Flower Recognition"). Empat kategori bunga yang digunakan:

| Kelas | Nama Ilmiah | Jumlah File | Format Dominan |
|-------|-------------|-------------|----------------|
| Bunga Matahari | Helianthus annuus | 99 | .jpg (97,9%) |
| Bunga Mawar | Rosa | 100 | .jpg (95%) |
| Bunga Melati | Jasminum | 98 | .jpg (94,9%) |
| Bunga Tulip | Tulipa | 100 | .jpg (100%) |
| **Total** | | **397** | **.jpg (96,9%)** |

### 5.2 Contoh Gambar Dataset

| Bunga Matahari | Bunga Mawar | Bunga Melati | Bunga Tulip |
|:---:|:---:|:---:|:---:|
| ![Matahari](assets/sample_matahari.jpg) | ![Mawar](assets/sample_mawar.png) | ![Melati](assets/sample_melati.jpg) | ![Tulip](assets/sample_tulip.jpg) |

### 5.3 Distribusi Data

Dataset memiliki distribusi yang **seimbang** antar kelas (selisih maksimal 2 gambar).

![Distribusi per Kelas](assets/image1.png)

**Gambar 1.** Distribusi Jumlah Gambar per Kelas

![Proporsi Dataset](assets/image2.png)

**Gambar 2.** Proporsi Dataset per Kelas

### 5.4 Analisis Resolusi dan Format

Resolusi gambar sangat bervariasi (236x211 hingga 3840x5760 piksel). Variasi tinggi ini menuntut proses **resizing** yang krusial pada tahap preprocessing.

![Sebaran Resolusi](assets/image3.png)

**Gambar 3.** Sebaran Resolusi Gambar per Kelas

Format JPEG mendominasi (96,9%). Format lain: PNG (3 file), WebP (3 file), GIF (1 file).

![Distribusi Format](assets/image5.png)

**Gambar 4.** Distribusi Format File

### 5.5 Analisis Warna

Setiap kelas memiliki karakteristik warna berbeda:
- **Matahari**: Dominan kuning (R dan G tinggi)
- **Mawar**: Dominan merah (R tinggi)
- **Melati**: Cenderung hijau/putih (G tinggi)
- **Tulip**: Variasi warna lebih beragam

![Sebaran Warna RGB](assets/image6.png)

**Gambar 5.** Sebaran Warna dalam Ruang RGB

![Rata-rata RGB per Kelas](assets/image7.png)

**Gambar 6.** Rata-rata Intensitas RGB per Kelas

### 5.6 Data Preparation

| Tahapan | Detail |
|---------|--------|
| **Filter Format** | Hanya .jpg, .jpeg, .png |
| **Konversi Warna** | RGBA/P → RGB (3 channel) |
| **Normalisasi** | Rescaling pixel ke [0,1] (pixel / 255.0) |
| **Augmentasi** | Rotation ±40°, Shift 20%, Shear 20%, Zoom 20%, Horizontal Flip |
| **Split Data** | 80% Train / 20% Validation (Stratified, random_state=42) |
| **Input Size** | 150x150 piksel |

Detail split dataset:

| Split | Matahari | Mawar | Melati | Tulip | Total |
|-------|----------|-------|--------|-------|-------|
| **Train** | 78 | 79 | 76 | 80 | **313** |
| **Validation** | 20 | 20 | 20 | 20 | **80** |

---

## 6. Model yang Digunakan

Proyek ini menggunakan **Transfer Learning** dengan backbone **MobileNetV2** (pretrained ImageNet). Berikut arsitektur lengkapnya:

### Arsitektur MobileNetV2

```
Input(150, 150, 3)
    ↓
MobileNetV2 (frozen, weights='imagenet', include_top=False)
    ↓  feature map 7x7x1280
GlobalAveragePooling2D()
    ↓
BatchNormalization()
    ↓
Dropout(0.3)
    ↓
Dense(128, activation='relu', kernel_regularizer=l2(0.001))
    ↓
BatchNormalization()
    ↓
Dropout(0.3)
    ↓
Dense(4, activation='softmax')
```

### Kode Implementasi

```python
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

### Callbacks

```python
callbacks = [
    ModelCheckpoint('models/flower_cnn.keras',
                    monitor='val_accuracy', save_best_only=True),
    EarlyStopping(patience=15, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
]
```

### Hyperparameter

| Parameter | Nilai |
|-----------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss Function | Categorical Crossentropy |
| Batch Size | 32 |
| Epochs | 50 (dengan Early Stopping) |
| Regularisasi | L2 (0.001) + Dropout (0.3-0.5) |
| LR Scheduler | ReduceLROnPlateau |

---

## 7. Hasil Pengujian

### 7.1 Training History

Berikut grafik akurasi dan loss selama pelatihan model MobileNetV2:

![Training History](assets/training_history.png)

**Gambar 7.** Training History - Akurasi dan Loss

### 7.2 Confusion Matrix

![Confusion Matrix](assets/confusion_matrix.png)

**Gambar 8.** Confusion Matrix - Hasil Evaluasi Model

### 7.3 Metrik Evaluasi

| Metrik | MobileNetV2 | ViT-Base/16 |
|--------|-------------|-------------|
| **Akurasi** | Baik (validation) | 100% (test set) |
| **Precision** | Tidak tercatat | 1.00 |
| **Recall** | Tidak tercatat | 1.00 |
| **F1-Score** | Tidak tercatat | 1.00 |
| **Epoch Training** | ~8-15 epoch | 8 epoch |
| **Waktu Training** | Cepat (CPU-friendly) | ±16,5 menit (GPU T4) |

---

## 8. Perbandingan Model

### 8.1 Tabel Perbandingan Detail

| Aspek | MobileNetV2 (TF) | ViT-Base/16 (PyTorch) |
|-------|-----------------|----------------------|
| **Total Parameter** | ± 2,4 juta | ± 86 juta |
| **Trainable Param** | ± 126 ribu | ± 86 juta |
| **Framework** | TensorFlow 2.x / Keras | PyTorch 2.x + timm |
| **Input Size** | 150x150 | 224x224 |
| **Pretrained** | ImageNet (1K classes) | ImageNet-21K (21K classes) |
| **Mekanisme** | Depthwise Separable Conv | Multi-Head Self-Attention |
| **Optimizer** | Adam (lr=0.001) | AdamW (lr=2e-5) |
| **Batch Size** | 32 | 16 |
| **Regularisasi** | L2 + Dropout | Weight Decay |
| **LR Scheduler** | ReduceLROnPlateau | CosineAnnealingLR |
| **Kebutuhan GPU** | Opsional (bisa CPU) | Wajib (GPU disarankan) |

### 8.2 Analisis Perbandingan

| Aspek | MobileNetV2 | ViT-Base |
|-------|-------------|----------|
| **Akurasi** | Baik, overfitting terkendali | Sempurna (100%), tapi potensi overfitting tinggi |
| **Efisiensi** | Sangat efisien (126K parameter) | Sangat berat (86M parameter) |
| **Kecepatan** | Cepat, bisa di CPU | Lambat, butuh GPU |
| **Dataset Kecil** | Cocok (risiko overfitting rendah) | Rentan overfitting |
| **Deployment** | Mobile/Edge device friendly | Butuh server bertenaga |

### 8.3 Kesimpulan Perbandingan

- **MobileNetV2** unggul dalam efisiensi dan cocok untuk deployment di perangkat terbatas.
- **ViT-Base** unggul akurasi namun但有 risiko overfitting pada dataset kecil dan butuh komputasi besar.
- Untuk dataset ±400 gambar, **MobileNetV2** lebih direkomendasikan karena trade-off yang lebih baik.

---

## 9. Referensi

[1] Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR*.

[2] Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *CVPR* (pp. 4510-4520).

[3] Howard, A. G., et al. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. *arXiv:1704.04861*.

[4] Krizhevsky, A., et al. (2012). ImageNet Classification with Deep Convolutional Neural Networks. *NeurIPS* (pp. 1097-1105).

[5] He, K., et al. (2016). Deep Residual Learning for Image Recognition. *CVPR* (pp. 770-778).
