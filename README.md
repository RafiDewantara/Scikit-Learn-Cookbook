# Scikit-Learn Cookbook

Repository ini berisi rangkuman, penjelasan teori, dan implementasi menggunakan Python dengan **scikit-learn** yang mengacu pada buku ***scikit-learn Cookbook: Over 80 Recipes for Machine Learning in Python with scikit-learn, Third Edition*** karya John Sukup, yang diterbitkan oleh Packt Publishing pada Desember 2025.

Repository ini disusun berdasarkan bab. Setiap bab membahas konsep, teknik, dan workflow machine learning, disertai contoh penerapan menggunakan Python dan scikit-learn.

Untuk saat ini, repository ini mencakup **Chapter 1–13**.

## Repository Contents

### Chapter 1 — Common Conventions and API Elements of scikit-learn

Bab ini memperkenalkan filosofi desain dan elemen utama API scikit-learn. Pemahaman terhadap struktur API membantu pengguna membangun workflow machine learning yang konsisten dan mudah dikembangkan.

Pada bab ini dibahas:

- Design philosophy of scikit-learn
- Estimators dan metode `fit()` serta `predict()`
- Transformers dan metode `transform()`
- Custom estimators dan transformers
- Pipelines dan workflow automation
- Common attributes dan methods
- Hyperparameter tuning menggunakan search methods
- Metadata, tags, dan metadata routing
- Best practices for API usage

Salah satu konsep utama scikit-learn adalah penggunaan **estimator**, yaitu objek yang mempelajari parameter dari data melalui `fit()`. Transformer dapat mengubah data menggunakan `transform()`, sedangkan estimator untuk supervised learning umumnya menyediakan `predict()` untuk menghasilkan prediksi.

Bab ini juga membahas **Pipeline**, yang menggabungkan preprocessing dan model ke dalam satu workflow. Penggunaan pipeline membantu menjaga konsistensi proses training dan prediction serta mengurangi risiko data leakage ketika dikombinasikan dengan cross-validation.

---

### Chapter 2 — Pre-Model Workflow and Data Preprocessing

Bab ini membahas tahapan persiapan data sebelum model machine learning dibangun. Kualitas data mentah dan cara preprocessing dapat memengaruhi performa serta kemampuan generalisasi model.

Pada bab ini dibahas:

- The impact of raw data on model performance
- Common data issues
- Cleaning dan preparing data
- Handling missing data
- Scaling techniques
- Encoding categorical variables
- Introduction to pipelines in scikit-learn
- Visualizing pipelines
- Feature engineering
- Practical exercises on data preprocessing

Preprocessing mencakup penanganan nilai yang hilang, pembersihan data, penyesuaian skala fitur, dan pengubahan variabel kategorikal ke representasi yang dapat diproses model. Pemilihan teknik perlu disesuaikan dengan karakteristik dataset dan kebutuhan algoritma.

Bab ini juga memperkenalkan pipeline untuk menggabungkan beberapa langkah preprocessing dengan estimator. Selain itu, **feature engineering** digunakan untuk membentuk atau mentransformasikan fitur agar informasi yang relevan dapat dimanfaatkan oleh model.

---

### Chapter 3 — Dimensionality Reduction Techniques

Bab ini membahas teknik **dimensionality reduction**, yaitu pendekatan untuk mengurangi jumlah fitur atau dimensi data sambil mempertahankan informasi yang relevan. Pengurangan dimensi dapat membantu visualisasi, mengurangi kompleksitas, dan mendukung proses pemodelan.

Pada bab ini dibahas:

- Introduction to dimensionality reduction
- Why dimensionality reduction is essential
- Theoretical foundations of dimensionality reduction
- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)
- Differences between PCA and LDA
- t-distributed Stochastic Neighbor Embedding (t-SNE)
- Data visualization
- Guidelines for selecting dimensionality reduction techniques
- Impact on model performance dan trade-offs
- Practical exercises using PCA and t-SNE

**PCA** membentuk principal components sebagai kombinasi fitur yang menangkap variasi data. PCA merupakan teknik unsupervised dan tidak menggunakan label kelas dalam pembentukan komponen.

Berbeda dengan PCA, **LDA** merupakan teknik supervised yang mempertimbangkan label kelas untuk mencari representasi yang memaksimalkan pemisahan antarkelas. Sementara itu, **t-SNE** terutama digunakan untuk memvisualisasikan struktur lokal data berdimensi tinggi ke ruang berdimensi lebih rendah.

Bab ini membahas pemilihan teknik berdasarkan tujuan analisis, karakteristik dataset, dan dampaknya terhadap performa model.

---

### Chapter 4 — Building Models with Distance Metrics and Nearest Neighbors

Bab ini memperkenalkan model berbasis jarak, khususnya **K-Nearest Neighbors (KNN)**. KNN menggunakan kedekatan antarobservasi untuk menentukan kelas atau nilai prediksi.

Pada bab ini dibahas:

- Introduction to distance metrics
- Understanding K-Nearest Neighbors (KNN)
- Distance metrics overview
- Hyperparameter tuning in KNN
- Evaluating KNN performance
- Evaluation metrics
- Practical exercises with KNN models
- Building a KNN classifier
- Tuning hyperparameters with grid search

KNN menentukan tetangga terdekat berdasarkan metrik jarak yang dipilih. Beberapa metrik jarak dapat memberikan hasil berbeda, sehingga pemilihannya perlu mempertimbangkan tipe fitur dan karakteristik data.

Nilai **K** merupakan hyperparameter penting. Nilai K yang terlalu kecil dapat membuat model sensitif terhadap noise, sedangkan nilai yang terlalu besar dapat menghaluskan batas keputusan secara berlebihan. Karena itu, tuning dan evaluasi diperlukan untuk memilih konfigurasi yang sesuai.

---

### Chapter 5 — Linear Models and Regularization

Bab ini membahas linear models untuk memodelkan hubungan antara fitur dan target, serta regularization untuk mengendalikan kompleksitas model dan mengurangi risiko overfitting.

Pada bab ini dibahas:

- Introduction to linear models
- Ridge regression
- Lasso regression
- ElasticNet
- Regularization theory and practice
- Regression and regularization
- Practical exercises with regularization techniques

Linear models membentuk prediksi berdasarkan kombinasi linear dari fitur. Pada regression, model berusaha memperkirakan nilai target numerik berdasarkan koefisien setiap fitur.

**Ridge regression** menggunakan regularisasi L2 yang mengecilkan nilai koefisien tanpa secara umum menjadikannya tepat nol. **Lasso regression** menggunakan regularisasi L1 yang dapat membuat sebagian koefisien menjadi nol sehingga juga dapat berperan dalam feature selection. **ElasticNet** menggabungkan penalti L1 dan L2.

Regularization membantu mengendalikan kompleksitas model. Pemilihan kekuatan regularisasi perlu dilakukan melalui evaluasi dan tuning agar model tidak terlalu sederhana maupun terlalu menyesuaikan diri dengan data training.

---

### Chapter 6 — Advanced Logistic Regression and Extensions

Bab ini memperluas pembahasan logistic regression untuk menangani berbagai kebutuhan classification, termasuk klasifikasi multiclass dan multilabel, regularisasi, serta evaluasi model.

Pada bab ini dibahas:

- Overview of logistic regression
- Multiclass classification techniques
- Regularization in logistic regression
- Multilabel classification concepts
- Model evaluation metrics
- Practical exercises with advanced logistic regression

**Logistic regression** digunakan untuk memodelkan probabilitas suatu observasi termasuk ke dalam kelas tertentu. Untuk kasus multiclass, strategi klasifikasi digunakan untuk menangani target yang memiliki lebih dari dua kategori. Pada multilabel classification, satu observasi dapat memiliki beberapa label sekaligus.

Regularization dapat diterapkan untuk mengendalikan besarnya koefisien dan membantu mengurangi overfitting. Bab ini juga membahas metrik evaluasi untuk menilai hasil classification model, sehingga pemilihan model tidak hanya bergantung pada satu ukuran performa.

---

### Chapter 7 — Support Vector Machines and Kernel Methods

Bab ini membahas **Support Vector Machines (SVM)**, yaitu algoritma yang membangun decision boundary dengan mencari pemisah antarkelas yang memiliki margin optimal. SVM juga dapat digunakan untuk regression.

Pada bab ini dibahas:

- Introduction to SVMs
- Kernel functions and their applications
- Tuning SVM parameters
- SVMs in high-dimensional spaces
- Evaluating SVM models
- Practical exercises with SVMs
- Building a simple SVM classifier
- Tuning SVM parameters with grid search
- Visualizing SVM decision boundaries

SVM berusaha menemukan hyperplane yang memisahkan kelas dengan margin sebesar mungkin. Untuk data yang tidak dapat dipisahkan secara linear, **kernel functions** memungkinkan SVM membentuk decision boundary non-linear melalui perhitungan pada ruang fitur yang lebih tinggi.

Performa SVM dipengaruhi oleh pemilihan kernel dan hyperparameter, seperti parameter regularisasi dan parameter kernel. Oleh karena itu, tuning serta evaluasi diperlukan untuk memperoleh model yang sesuai dengan karakteristik data.

---

### Chapter 8 — Tree-Based Algorithms and Ensemble Methods

Bab ini membahas algoritma berbasis decision tree dan metode ensemble yang menggabungkan beberapa model untuk menghasilkan prediksi yang lebih kuat atau stabil.

Pada bab ini dibahas:

- Introduction to decision trees
- Random forests and bagging
- Gradient boosting machines
- Hyperparameter tuning for trees and ensembles
- Comparing ensemble methods
- Practical exercises with tree-based models

**Decision tree** membentuk aturan keputusan melalui pemisahan data berdasarkan fitur. Struktur pohon relatif mudah diinterpretasikan, tetapi pohon yang terlalu kompleks dapat mengalami overfitting.

**Random forest** membangun banyak decision trees menggunakan pendekatan bagging dan menggabungkan hasil prediksinya. **Gradient boosting** membangun model secara bertahap, dengan model berikutnya berusaha memperbaiki kesalahan model sebelumnya.

Bab ini membahas tuning hyperparameter dan perbandingan metode ensemble untuk memahami trade-off performa, kompleksitas, dan generalisasi model.

---

### Chapter 9 — Text Processing and Multiclass Classification

Bab ini membahas pemrosesan data teks agar dapat digunakan sebagai input machine learning models, serta penerapan classification untuk target dengan banyak kelas.

Pada bab ini dibahas:

- Introduction to text processing
- Text vectorization techniques
- Feature extraction from text
- Implementing text classification models
- Multiclass classification strategies
- Evaluating text models
- Practical exercises in text processing
- Preprocessing and vectorizing text
- Feature extraction with n-grams
- Building and evaluating a multiclass classifier

Data teks perlu diubah menjadi representasi numerik sebelum digunakan oleh sebagian besar algoritma machine learning. **Text vectorization** dan **feature extraction** mengubah dokumen menjadi fitur yang dapat diproses model. Salah satu pendekatan yang dibahas adalah penggunaan n-grams untuk merepresentasikan rangkaian kata atau token.

Setelah fitur teks dibentuk, classification models dapat digunakan untuk memprediksi kategori dokumen. Bab ini juga membahas strategi multiclass classification dan evaluasi model teks.

---

### Chapter 10 — Clustering Techniques

Bab ini membahas **clustering**, yaitu teknik unsupervised learning untuk mengelompokkan observasi berdasarkan kemiripan karakteristik tanpa menggunakan label target.

Pada bab ini dibahas:

- Introduction to clustering
- K-means clustering
- Hierarchical clustering
- Density-based clustering with DBSCAN
- Cluster evaluation metrics
- Choosing the right clustering algorithm
- Advanced clustering techniques
- Practical exercises with clustering models

**K-means** membagi data ke dalam sejumlah cluster berdasarkan kedekatan observasi terhadap centroid. **Hierarchical clustering** membentuk struktur pengelompokan bertingkat yang dapat divisualisasikan menggunakan dendrogram.

**DBSCAN** mengelompokkan data berdasarkan kepadatan area dan dapat mengidentifikasi observasi yang tidak termasuk cluster. Setiap algoritma memiliki karakteristik dan asumsi yang berbeda, sehingga pemilihan metode perlu mempertimbangkan bentuk cluster, kepadatan data, ukuran dataset, dan noise.

Bab ini juga membahas cluster evaluation metrics untuk membantu menilai kualitas hasil pengelompokan serta membandingkan algoritma clustering.

---

### Chapter 11 — Novelty and Outlier Detection

Bab ini membahas teknik untuk mengidentifikasi observasi yang berbeda secara signifikan dari pola umum data. **Outlier detection** mencari data yang tidak biasa dalam dataset, sedangkan **novelty detection** berfokus pada identifikasi observasi baru yang berbeda dari pola data normal yang telah dipelajari.

Pada bab ini dibahas:

- Introduction to outlier and novelty detection
- Isolation Forest
- One-Class Support Vector Machine
- Local Outlier Factor (LOF)
- Evaluating outlier detection models
- Handling detected outliers
- Choosing the right detection technique
- Practical exercises using real-world and sensor-related datasets

**Isolation Forest** mendeteksi anomali berdasarkan seberapa mudah suatu observasi terisolasi melalui pemisahan data secara acak. **One-Class SVM** mempelajari batas yang merepresentasikan area data normal. **Local Outlier Factor (LOF)** membandingkan kepadatan lokal suatu observasi dengan kepadatan tetangganya.

Hasil deteksi perlu dievaluasi dan ditafsirkan sesuai konteks. Tidak semua outlier merupakan kesalahan data; sebagian dapat menunjukkan kejadian penting, kondisi langka, atau potensi masalah. Karena itu, penanganan outlier perlu dilakukan berdasarkan tujuan analisis.

---

### Chapter 12 — Cross-Validation and Model Evaluation Techniques

Bab ini membahas metode evaluasi model dan **cross-validation** untuk memperkirakan kemampuan model dalam melakukan generalisasi terhadap data yang belum pernah dilihat.

Pada bab ini dibahas:

- Introduction to cross-validation
- Advanced cross-validation methods
- Model evaluation metrics overview
- Implementing cross-validation in scikit-learn
- Model selection techniques
- Evaluating model generalizability
- Practical exercises in cross-validation and evaluation
- Grid search
- Learning curves and validation curves

Cross-validation membagi data menjadi beberapa bagian atau folds. Model dilatih dan dievaluasi secara bergantian pada bagian data yang berbeda, sehingga evaluasi tidak hanya bergantung pada satu pembagian training dan testing.

Bab ini juga membahas penggunaan cross-validation dalam scikit-learn, pemilihan model, hyperparameter tuning, dan metrik evaluasi. **Learning curves** dan **validation curves** dapat membantu memahami hubungan antara jumlah data training, kompleksitas model, dan performa evaluasi.

---

### Chapter 13 — Deploying scikit-learn Models in Production

Bab ini membahas tahapan membawa model scikit-learn dari lingkungan eksperimen ke lingkungan production. Deployment tidak hanya mencakup penyimpanan model, tetapi juga skalabilitas, monitoring, pembaruan, dan pengelolaan siklus hidup model.

Pada bab ini dibahas:

- Overview of model deployment
- Serialization and persistence techniques
- Scaling models for production
- Monitoring and updating deployed models
- Managing the model life cycle
- Setting up deployment pipelines
- Practical exercises in model deployment

**Serialization and persistence** memungkinkan model yang telah dilatih disimpan dan dimuat kembali tanpa harus melakukan training dari awal. Model yang digunakan dalam production juga perlu mempertimbangkan kebutuhan scaling, integrasi ke aplikasi, dan konsistensi lingkungan eksekusi.

Setelah deployment, performa model perlu dipantau karena distribusi data dan pola penggunaan dapat berubah. Bab ini membahas monitoring, pembaruan model, pengelolaan model life cycle, dan deployment pipelines untuk mendukung proses deployment yang lebih terstruktur dan maintainable.

---

## Chapter Overview

Tiga belas bab dalam buku ini membentuk alur pembelajaran machine learning menggunakan scikit-learn, mulai dari pemahaman API, persiapan data, pemilihan algoritma, evaluasi, hingga deployment model.

| Chapter | Main Topic | Main Purpose |
| ------- | ---------- | ------------ |
| 1 | Common Conventions and API Elements of scikit-learn | Memahami filosofi desain dan penggunaan API scikit-learn |
| 2 | Pre-Model Workflow and Data Preprocessing | Membersihkan, menyiapkan, dan mentransformasikan data |
| 3 | Dimensionality Reduction Techniques | Mengurangi dimensi dan memvisualisasikan data |
| 4 | Building Models with Distance Metrics and Nearest Neighbors | Membangun model berbasis jarak dan KNN |
| 5 | Linear Models and Regularization | Membangun linear models dan mengendalikan kompleksitasnya |
| 6 | Advanced Logistic Regression and Extensions | Mengembangkan logistic regression untuk classification yang lebih kompleks |
| 7 | Support Vector Machines and Kernel Methods | Membangun model SVM dan menggunakan kernel methods |
| 8 | Tree-Based Algorithms and Ensemble Methods | Menggunakan decision trees dan ensemble learning |
| 9 | Text Processing and Multiclass Classification | Mengolah data teks dan membangun multiclass classifier |
| 10 | Clustering Techniques | Mengelompokkan data tanpa label |
| 11 | Novelty and Outlier Detection | Mengidentifikasi observasi tidak biasa dan data baru |
| 12 | Cross-Validation and Model Evaluation Techniques | Mengevaluasi model dan mengukur generalisasi |
| 13 | Deploying scikit-learn Models in Production | Menyimpan, menerapkan, memantau, dan mengelola model di production |

Secara keseluruhan, buku ini membahas workflow machine learning yang dimulai dari **memahami API scikit-learn dan menyiapkan data**, dilanjutkan dengan **reduksi dimensi serta pembangunan berbagai model supervised dan unsupervised learning**. Setelah model dibangun, proses dilanjutkan dengan **evaluasi, validasi, pemilihan model, dan deployment ke production**.

## Reference

Sukup, J. (2025). *scikit-learn Cookbook: Over 80 Recipes for Machine Learning in Python with scikit-learn* (3rd ed.). Packt Publishing.
