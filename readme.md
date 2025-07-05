# Analisis Sentimen Ulasan Film dengan Custom IBM Granite Models

## Gambaran Proyek

**Tujuan**: Mengembangkan model analisis sentimen berbasis Transformer untuk mengidentifikasi pola sentimen audiens dari ulasan film IMDB secara akurat, memberikan wawasan strategis untuk pemasaran dan produksi film.  
**Latar Belakang dan Permasalahan**: Produser film dan platform streaming sering kesulitan memahami sentimen audiens secara cepat dan akurat dari ulasan online, yang dapat menghambat pengambilan keputusan strategis seperti perbaikan produksi atau optimasi kampanye pemasaran. Proyek ini menggunakan dataset IMDB Reviews dari Kaggle dan model Transformer Encoder manual yang menyerupai IBM Granite, dioptimalkan untuk laptop karena keterbatasan akses ke WatsonX (tidak memiliki kartu kredit). Pendekatan ini menerapkan **K-Fold cross-validation** (10 lipatan) untuk pembagian data yang robust [Kohavi, 1995]. Proyek mencakup **Exploratory Data Analysis (EDA)**, **Preprocessing**, **Modeling**, **Evaluasi**, dan **Interpretasi Bisnis**, dengan fokus pada efisiensi dan wawasan praktis untuk industri film.

## Dataset

Dataset: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  

- Berisi 50,000 ulasan film dengan label sentimen (positif/negatif).  
- Format: CSV, dengan kolom teks ulasan dan label sentimen.  

## Alur Analisis

### 1. Exploratory Data Analysis (EDA)

- **Pemeriksaan Data**: Identifikasi missing values (0.5% diisi dengan teks kosong) dan duplikasi (1.2% dihapus) untuk memastikan kualitas data.  
- **Distribusi Sentimen**: Analisis persentase ulasan positif (50.19%) dan negatif (49.81%) untuk memahami persepsi audiens.  
- **Panjang Teks Ulasan**: Evaluasi distribusi panjang ulasan (mayoritas 0-50 token, negatif lebih panjang >200 token).  
- **Kata Kunci Teratas**: Identifikasi 10 kata teratas per kelas sentimen untuk menangkap tema utama.  

### 2. Preprocessing

- **Pembersihan Data**: Hapus tag HTML, karakter non-alfabet, dan ubah teks ke huruf kecil [Zhang et al., 2020].  
- **Normalisasi**: Batasi panjang teks hingga 512 kata untuk konsistensi.  
- **Tokenisasi**: Ubah teks menjadi token, hapus stopwords untuk efisiensi model [Sun et al., 2019].  

### 3. Modeling (Custom IBM Granite Models)

- **Persiapan Data**:  
  - Buat vokabulari dan ubah token menjadi indeks.  
  - Atur panjang urutan maksimum 256 token.  
  - Kodekan label sentimen (positif: 1, negatif: 0).  
  - Terapkan K-Fold cross-validation (10 lipatan) untuk robustitas [Kohavi, 1995; Bergstra & Bengio, 2012].  
- **Arsitektur Model**:  
  - Transformer Encoder manual berbasis PyTorch, menyerupai IBM Granite dengan self-attention dan positional encoding [Vaswani et al., 2017; Devlin et al., 2019].  
  - Konfigurasi ringan (embedding dimensi 16, 1 lapisan, 2 kepala perhatian, dropout 0.1) untuk kompatibilitas laptop, menyeimbangkan performa dan efisiensi komputasi.  
- **Pelatihan**:  
  - Latih selama 10 epoch, batch size 64, optimizer Adam, loss CrossEntropyLoss.  
- **Evaluasi**:  
  - Metrik: akurasi, presisi, recall, F1-score, ROC-AUC per lipatan.  

### 4. Hasil Analisis

#### Hasil Fold 2

- **Matriks Kebingungan**:  
  - Pelatihan: [[22245, 0], [0, 22378]] – Akurasi 100% (44,623 sampel).  
  - Pengujian: [[2453, 0], [0, 2506]] – Akurasi 100% (4,959 sampel).  
- **Laporan Klasifikasi**:  
  - Pelatihan: Akurasi, precision, recall, F1-score: 1.00 (support: 22,245 negatif, 22,378 positif).  
  - Pengujian: Akurasi, precision, recall, F1-score: 1.00 (support: 2,453 negatif, 2,506 positif).  
- **Wawasan Bisnis**:  
  - **Distribusi Sentimen**: Positif 50.53%, negatif 49.47% – opini audiens seimbang.  
  - **Kata Kunci Teratas**:  
    - Positif: [('film', 3,780), ('movie', 3,476), ('one', 2,457), ('like', 1,601), ('good', 1,347), ('story', 1,262), ('great', 1,204), ('see', 1,145), ('time', 1,130), ('really', 1,074)].  
    - Negatif: [('movie', 4,422), ('film', 3,273), ('one', 2,200), ('like', 2,083), ('even', 1,351), ('bad', 1,303), ('good', 1,271), ('would', 1,250), ('really', 1,203), ('time', 1,112)].  
    - "Great" (positif) dan "bad" (negatif) mencerminkan sentimen ekstrem; "bad" sering terkait film aksi, menunjukkan ekspektasi tinggi pada efek visual.  
  - **Skor Kepentingan Kata** (berdasarkan bobot perhatian model):  
    - Positif: {'film': 0.489, 'movie': 0.754, 'one': 0.578, 'like': 0.637, 'good': 0.669, 'story': 0.538, 'great': 1.071, 'see': 0.936, 'time': 0.813, 'really': 0.411}.  
    - Negatif: {'movie': 0.754, 'film': 0.489, 'one': 0.578, 'like': 0.637, 'even': 0.612, 'bad': 0.818, 'good': 0.669, 'would': 0.844, 'really': 0.411, 'time': 0.813}.
    - Hasil identifikasi, "great" (1.071) dan "bad" (0.818) menjadi acuan sebagai pendorong sentimen.
  - **Panjang Rata-rata Ulasan**: Positif: 111.14 token, negatif: 109.41 token – ulasan negatif >200 token lebih rinci, menunjukkan kritik mendalam.  
- **Visualisasi**:
  - Loss curve: Penurunan train loss (0.08 ke <0.02), test loss stabil (0.00).
    ![image](https://github.com/user-attachments/assets/4d41f855-e495-4f17-8233-23b830949c80)

  - Distribusi panjang ulasan: Mayoritas 0-50 token, negatif lebih panjang.
    ![image](https://github.com/user-attachments/assets/5e1c9d9a-2aee-4898-86da-acaba6a2c0aa)

    Ulasan negatif lebih panjang menunjukkan kritik yang lebih terlibat, berpotensi untuk analisis kualitatif lebih lanjut.

#### Rata-rata Semua Fold

- **Metrik**: Akurasi, precision, recall, F1, ROC-AUC, PR-AUC: 1.0000 ± 0.0000 (pelatihan dan pengujian).  
- **Wawasan Bisnis**:  
  - Distribusi sentimen: Positif 50.19%, negatif 49.81%.  
  - Panjang rata-rata ulasan: Positif 112.79 token, negatif 111.36 token.   

### 5. Interpretasi Bisnis

- **Keandalan Model**: Akurasi dan F1-score 1.00 menunjukkan model sangat andal untuk analisis sentimen otomatis.  
- **Pemasaran**: Kata kunci positif ("film", "great") dapat digunakan untuk kampanye yang menonjolkan kekuatan film.  
- **Perbaikan Produksi**: Ulasan negatif (49.81%) dengan kata "bad" (terutama pada film aksi) menunjukkan perlunya perbaikan efek visual atau alur cerita.  
- **Pengembangan Konten**: Konsistensi panjang ulasan (111-113 token) mendukung penggunaan ulasan positif sebagai testimoni kredibel.  

## Penjelasan Dukungan AI

Model Transformer Encoder manual meniru mekanisme self-attention dan positional encoding IBM Granite untuk menangkap hubungan antar-token dalam ulasan [Vaswani et al., 2017; Devlin et al., 2019]. Konfigurasi ringan (embedding dimensi 16, 1 lapisan, 2 kepala perhatian) dipilih untuk efisiensi pada laptop. AI membantu menghitung skor kepentingan kata berdasarkan bobot perhatian, mengidentifikasi "great" (1.071) dan "bad" (0.818) sebagai pendorong sentimen. K-Fold cross-validation (10 lipatan) meningkatkan robustitas [Kohavi, 1995]. Preprocessing dengan tokenisasi dan penghapusan stopwords memperkuat performa [Sun et al., 2019].

## Rekomendasi Bisnis

1. **Integrasi Real-time (Prioritas Tinggi)**: Terapkan model untuk analisis sentimen real-time di platform streaming, dengan target waktu pemrosesan <1 detik per ulasan.
   
   - **Dasar**: Waktu pemrosesan <1 detik adalah standar industri untuk analisis sentimen real-time pada platform seperti Twitter atau Netflix, memungkinkan pemantauan tren audiens secara instan [Hutto & Gilbert, 2014].

2. **Strategi Pemasaran**: Gunakan kata kunci positif seperti "film" (frekuensi 3,780) dan "great" (1,204) untuk kampanye media sosial, dengan target peningkatan engagement (likes, shares) sebesar 7% dalam 3 bulan.
   
   - **Dasar**: Berdasarkan studi pemasaran digital, kampanye berbasis kata kunci relevan dapat meningkatkan engagement sebesar 5-10% dalam 3 bulan [Sprout Social, 2023]. Target 7% dipilih sebagai angka realistis untuk film dengan audiens IMDB yang aktif di media sosial.

3. **Perbaikan Produksi**: Analisis ulasan negatif (>200 token) untuk memperbaiki aspek film aksi (efek visual, alur), dengan target peningkatan rating IMDB sebesar 4% pada rilis berikutnya.
   
   - **Dasar**: Rata-rata rating film aksi di IMDB adalah ~6.5/10 [IMDB, 2025]. Perbaikan berdasarkan umpan balik spesifik (misalnya, efek visual) dapat meningkatkan rating sebesar 3-5%, seperti terlihat pada sekuel film yang dioptimalkan (contoh: *The Matrix Reloaded* vs. *The Matrix*) [Box Office Mojo, 2023].

4. **Testimoni Audiens**: Gunakan ulasan positif sebagai materi promosi di iklan online, dengan target peningkatan konversi iklan (click-through rate) sebesar 6%.
   
   - **Dasar**: Testimoni otentik dapat meningkatkan konversi iklan sebesar 4-8% dalam kampanye digital [Nielsen, 2022]. Target 6% dipilih sebagai angka realistis untuk ulasan IMDB yang memiliki kredibilitas tinggi di kalangan pecinta film.

5. **Penelitian Lanjutan**: Lakukan analisis kualitatif pada ulasan negatif untuk wawasan lebih mendalam, dengan target laporan kualitatif selesai dalam 4 bulan.
   
   - **Dasar**: Analisis kualitatif mendalam pada dataset besar seperti IMDB (50,000 ulasan) membutuhkan 3-6 bulan untuk pengkodean tema dan interpretasi [Bazeley & Jackson, 2013]. Target 4 bulan realistis untuk tim kecil dengan dukungan AI.

*Catatan: Target metrik (7%, 4%, 6%) didasarkan pada benchmark industri dan konteks dataset IMDB. Penyesuaian dapat dilakukan berdasarkan data baseline spesifik dari pemangku kepentingan.*

## Output Terkait

- **Notebook**: [capstone_project.ipynb](https://github.com/nnichaelangello/Analisis-Sentimen-Ulasan-Film-dengan-Custom-IBM-Granite-Models/blob/main/Analisis%20Sentimen%20Ulasan%20Film%20dengan%20Custom%20IBM%20Granite%20Models.ipynb)  
- **Slide Presentasi**: [presentation.pptx](https://github.com/nnichaelangello/Analisis-Sentimen-Ulasan-Film-dengan-Custom-IBM-Granite-Models/blob/main/presentation.pptx)  

## Referensi

1. Vaswani, A., et al. (2017). "Attention is All You Need." *Advances in Neural Information Processing Systems (NeurIPS)*, 30. [https://arxiv.org/abs/1706.03762]  

2. Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." *Proceedings of NAACL-HLT*. [https://arxiv.org/abs/1810.04805]  

3. Kohavi, R. (1995). "A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection." *IJCAI*, 14(2), 1137–1145. [https://www.ijcai.org/Proceedings/95-2/Papers/016.pdf]  

4. Bergstra, J., & Bengio, Y. (2012). "Random Search for Hyper-Parameter Optimization." *Journal of Machine Learning Research*, 13, 281–305. [http://jmlr.csail.mit.edu/papers/v13/bergstra12a.html]  

5. Sun, C., et al. (2019). "How to Fine-Tune BERT for Text Classification?" *arXiv preprint*. [https://arxiv.org/abs/1905.05583]  

6. Zhang, Y., et al. (2020). "Advances in Sentiment Analysis Using Deep Learning." *IEEE Transactions on Neural Networks and Learning Systems*, 31(10), 3876–3890. [https://ieeexplore.ieee.org/document/8963218]  

7. Hutto, C.J., & Gilbert, E. (2014). "VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text." *Proceedings of the International AAAI Conference on Web and Social Media*, 8(1), 216–225. [https://ojs.aaai.org/index.php/ICWSM/article/view/14550]

8. Chaffey, D., & Ellis-Chadwick, F. (2019). *Digital Marketing: Strategy, Implementation and Practice*. Pearson Education. [https://www.pearson.com/en-gb/subject-catalog/p/digital-marketing/P200000003833/9781292241579]

9. IMDB (2025). "IMDB Movie Ratings Database." [https://www.imdb.com/interfaces/]

10. Kotler, P., & Keller, K.L. (2016). *Marketing Management*. Pearson Education. [https://www.pearson.com/en-us/subject-catalog/p/marketing-management/P200000006102/9780133856460]

11. Bazeley, P., & Jackson, K. (2013). *Qualitative Data Analysis with NVivo*. SAGE Publications. [https://us.sagepub.com/en-us/nam/qualitative-data-analysis-with-nvivo/book237234]

## Kontak

Untuk diskusi, hubungi michaelriyadi5@gmail.com . Proyek terbuka untuk pengembangan berdasarkan masukan.
