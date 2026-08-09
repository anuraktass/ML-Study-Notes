# ML-Study-Notes
ML/DL Algorithms Cheat Sheet: Random Forest, SVM, XGBoost, LightGBM, CNN, RNN, LSTM, BERT, GPT, PCA, ARIMA, Reinforcement Learning + Exam Notes


# 🤖 Makine Öğrenmesi & Yapay Zeka Ansiklopedisi

> Tarih: Ağustos 2026  
> Hazırlayan: [Emine Nur AKTAŞ]

Bu tablo, Makine Öğrenmesi sınavıma hazırlanırken tüm konuları **tek bir pencereden** görebilmek için oluşturduğum kapsamlı notlarımdır. 🚀

---

##  TÜM MAKİNE ÖĞRENMESİ EVRENİ (EKSİKSİZ BÜYÜK TABLO)

| Kategori | İçindeki Modeller / Yöntemler |  Akılda Kalacak Tek Cümle |  Hangi Sorunu Çözer? | ipuçlari |
| :--- | :--- | :--- | :--- | :--- |
| ** VERİ HAZIRLIK** | Numpy, Pandas, Scikit-Learn | Veriyi bilgisayarın anlayacağı sayısal sofraya hazırlar. | Eksik veriyi doldurmak, kategorik veriyi sayıya çevirmek (One-Hot Encoding). | Pandas = Tablo, Numpy = Matematik. İkisi birlikte çalışır. |
| ** REGRESYON (Sayı Tahmin)** | Lineer, Ridge, Lasso, Elastic Net, SVR | Bir sayı kestirir (ev fiyatı, sıcaklık). | Tahmin işleri. | Lasso gereksiz özellikleri eler (sıfır yapar). Ridge hepsini biraz küçültür. |
| ** KLASİK SINIFLANDIRMA** | Lojistik Regresyon, SVM, KNN, Naive Bayes | Veriyi etiketli gruplara ayırır (kedi/köpek, spam/değil). | Elinizde etiketli örnekler varsa. | SVM'de "Kernel" veriyi yukarı taşır. KNN komşulara bakar. |
| ** TOPLULUK (Ensemble)** | Random Forest, AdaBoost, Gradient Boosting (XGBoost, LightGBM, CatBoost) | Tek ağaç zayıf, bir orman (RF) veya ardışık hata düzeltme (Boosting) ile güçlenir. | En yüksek doğruluğun istendiği her yer (Kaggle yarışmaları). | RF = Paralel (aynı anda), Boosting = Ardışık (sırayla hata düzeltir). LightGBM çok hızlıdır. |
| ** DENETİMSİZ (Kümeleme)** | K-Means, Hiyerarşik, DBSCAN, GMM | Etiketsiz veriyi kendi içinde doğal gruplara ayırır. | Müşteri segmentasyonu, ürün gruplama. | K-Means yuvarlak kümeler bulur, DBSCAN ise şekilsiz kümeler ve aykırıları da bulur. |
| ** ANOMALİ (Aykırı Tespit)** | İzolasyon Ormanı, One-Class SVM, Z-Score, IQR | Çoğunluğa uymayan sapkın verileri yakalar. | Kredi kartı sahtekarlığı, makine arıza tespiti. | Aykırılar her zaman kötü değildir; bazen hırsızı yakalamak için aranır! |
| ** BOYUT AZALTMA** | PCA, t-SNE, LDA | Çok fazla sütun (özellik) varsa, bunları özetleyerek azaltır. | Gereksiz değişkenlerden kurtulmak, veriyi 2D/3D'de görselleştirmek. | PCA sayısal veriyi sıkıştırır. t-SNE sadece görselleştirme içindir. |
| ** DOĞAL DİL İŞLEME (NLP)** | Bag of Words, TF-IDF, Word2Vec, BERT, GPT, RNN, LSTM | İnsan dilini (yazı/metin) bilgisayarın anlayacağı sayılara çevirir. | Metin sınıflandırma (duygu analizi), chatbot, çeviri, arama motorları. | TF-IDF = Kelimenin önem skoru. BERT/GPT = Derin öğrenme ile dil anlama (Transformer). |
| ** ZAMAN SERİSİ** | ARIMA, SARIMA, Prophet, LSTM, GARCH | Zamana bağlı (tarih/saat) verileri analiz eder ve ileriyi tahmin eder. | Hava durumu, hisse senedi fiyatı, mağaza satış tahmini. | ARIMA istatistikseldir, LSTM derin öğrenmeli uzun vadeli hafızaya sahiptir. |
| ** GÖRÜNTÜ İŞLEME** | CNN, OpenCV, YOLO, GAN | Resimleri piksel piksel inceleyerek desen ve nesne tanır. | Yüz tanıma, otonom araba (yaya), tıbbi görüntü (MR). | CNN = Evrişim (filtre) ile özellik çıkarır. GAN ise yeni sahte resim üretir. |
| ** PEKİŞTİRMELİ (RL)** | Q-Learning, DQN, Policy Gradient, PPO | Ödül/ceza sistemiyle deneyerek öğrenir (oyun oynayarak öğrenmek gibi). | Oyun oynama (AlphaGo), robot kontrolü, otonom araç yönlendirme. | Etiketi yoktur! Ajan (agent) ortamla etkileşime girerek öğrenir. |
| ** MODEL DOĞRULAMA** | Train/Test Split, Cross-Validation (K-Fold) | Modeli sınamak için veriyi eğitim ve test olarak böler. | Model gerçekten öğrendi mi, yoksa ezberledi mi? (Overfitting kontrolü). | Cross-Validation veriyi katlara böler, daha güvenilir sonuç verir. |
| ** OPTİMİZASYON** | Gradient Descent, Adam, RMSprop | Modelin hatasını (Loss) en aza indiren matematiksel iniş yöntemi. | Tüm modellerin eğitimi sırasında kullanılan arka plan motoru. | Learning Rate (α) çok büyük = sıçrar kaçar, çok küçük = çok yavaş ilerler. |
| ** TRANSFER ÖĞRENME** | Fine-Tuning, Feature Extraction (VGG, ResNet, BERT) | Önceden eğitilmiş büyük bir modeli alıp, kendi küçük verinize uyarlamak. | Az veriniz varsa ve büyük modeli sıfırdan eğitemiyorsanız. | Zaman ve veri kazandırır. Görüntüde ResNet, dilde BERT popülerdir. |
| ** OTOMATİK (AutoML)** | Grid Search, Random Search, Bayesian | Modelin en iyi ayarlarını (parametrelerini) otomatik olarak arar. | En iyi performansı bulmak için manuel denemek yerine makineye aramak. | Grid Search = Tüm kombinasyonları dener (yavaş). Random Search = Rastgele dener (hızlı). |


## ⚠️ En Çok Karıştırılan 5 İkili 
- **Random Forest vs Boosting:** RF paralel (aynı anda), Boosting ardışık (sırayla).
- **Ridge vs Lasso:** Ridge hepsini küçültür, Lasso bazılarını SIFIR yapar (özellik seçer).
- **CNN vs RNN:** CNN resim içindir (uzamsal), RNN zaman/ses içindir (zamansal).
- **Unsupervised vs Anomaly:** Unsupervised KÜME bulur, Anomaly ise AYKIRI (sapkın) bulur.
- **Regresyon vs Sınıflandırma:** Regresyon çıktısı SAYI, Sınıflandırma çıktısı ETİKET/SINIF.

---

