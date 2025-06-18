
# SecurityAI 🔐🧠

## 📌 Proje Hakkında

SecurityAI, yapay zeka tabanlı bir güvenlik sistemidir. Proje, yüz tanıma ve duygu analizi tekniklerini kullanarak potansiyel tehdit oluşturan bireyleri tespit etmeyi amaçlamaktadır. Kamera görüntülerinden elde edilen veriler, AI modelleriyle analiz edilerek güvenlik birimlerine gerekli bildirimler yapılır.

## 🧠 Kullanılan Teknolojiler

- 🖥️ Python
- 🤖 TensorFlow / PyTorch
- 📷 OpenCV (kamera görüntü işleme)
- 😐 Duygu analizi modeli (Emotion Classification)
- 🧬 Yüz tanıma modeli (Face Recognition)
- 🔧 Google Colab & Kaggle entegrasyonu

## 🚀 Özellikler

- Gerçek zamanlı yüz tanıma (Real-time face recognition)
- 8 temel duygu üzerinden duygu analizi: Happiness, Sadness, Anger, Fear, Disgust, Surprise, Neutral, Love
- Tehlikeli ruh hali veya tanımlanamayan kişi algılandığında alarm tetikleme
- Google Colab üzerinde eğitim ve test imkanı
- TensorBoard ile eğitim süreci takibi
- Kullanıcıya özel veri tanıtımı ve model güncelleme desteği

## ⚙️ Kurulum ve Çalıştırma

1. Bu repoyu klonlayın:
   ```bash
   git clone https://github.com/innery/SecurityAI.git
   ```

2. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

3. Eğitim dosyasını çalıştırın:
   ```bash
   python train_model.py
   ```

4. Gerçek zamanlı test için:
   ```bash
   python run_camera.py
   ```

> Not: Model ağırlıklarını ve veri kümesini `models/` ve `dataset/` klasörlerine yerleştirmeyi unutmayın.

## 📁 Proje Yapısı

```
SecurityAI/
├── models/             # Eğitimli yüz ve duygu tanıma modelleri
├── dataset/            # Sentetik ve gerçek yüz/durum görselleri
├── train_model.py      # Model eğitimi
├── run_camera.py       # Kamera ile gerçek zamanlı test
├── utils.py            # Yardımcı fonksiyonlar
└── requirements.txt    # Gerekli kütüphaneler
```

## 📄 Lisans

Bu proje yalnızca akademik ve kişisel kullanım içindir. Ticari kullanım için geliştirici izni gereklidir.
