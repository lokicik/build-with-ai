# 🚀 Build With AI: Görüntü İşleme Eğitimi

Bu repo, 26 Şubat 2025 tarihinde Lokman Baturay Efe ve Niyazi Mert Işıksal tarafından Mühendislik Fakültesi D101'de düzenlenen "Build With AI: Görüntü İşleme Eğitimi" etkinliğinde kullanılan örnek kodları ve materyalleri içermektedir.

## 📋 Etkinlik Hakkında

GDG On Campus Trakya University bünyesinde düzenlenen bu eğitimde, görüntü işlemenin temellerinden ileri seviye uygulamalarına kadar geniş bir yelpazede konular ele alındı:

### 💡 Eğitimde Neler Yaptık?

- ✅ Görmek, görsel, imaj, resim ve görüntü kavramlarını tarihsel ve dilbilimsel olarak ele aldık.
- ✅ Çözünürlük, 1080p'deki p ne anlama gelir, PPI ve DPI gibi kavramlarından bahsettik.
- ✅ Görüntü işlemenin tarihine yolculuk yaparak, New York - Londra arasındaki ilk görüntü iletim sistemini inceledik.
- ✅ Günlük hayatta görüntü işlemenin kullanım alanlarını konuştuk.
- ✅ OpenCV ve OCR teknolojilerine odaklanarak, temel görüntü işleme algoritmalarını uyguladık:
  - 🔹 Resim döndürme, çözünürlük ayarlama, kontür çıkarma, thresholding
  - 🔹 Renk kanalı değiştirme
  - 🔹 Kenar ve yüz tanıma
- ✅ Görüntü işleyebilen LLM'lerin OCR modellerinin yerini alıp alamayacağını tartıştık.

### 🎯 Derin Öğrenme Bölümü

- 🔍 CNN (Convolutional Neural Networks) algoritmaları:
  - 🔹 Konvolüsyon, pooling ve fully connected katmanları
  - 🔹 Görüntü işleme adımları, matris işlemleri ve aktivasyon fonksiyonları
- 🤖 Roboflow üzerinden model geliştirme:
  - ✅ Veri etiketleme ve topluluk veri setleriyle çalışma
  - ✅ Model eğitimi ve kullanımı
  - ✅ Marvel karakterlerini tanıyabilen bir model oluşturma

## 🛠️ Kurulum

Gerekli kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
```

## 📚 İçerik

1. `01_basics.py` - OpenCV'ye giriş ve temel işlemler
2. `02_image_operations.py` - Görüntü işleme operasyonları (yeniden boyutlandırma, kırpma, döndürme)
3. `03_color_spaces.py` - Renk uzayları ve dönüşümleri
4. `04_drawing.py` - Şekil çizme ve metin ekleme
5. `05_thresholding.py` - Eşikleme teknikleri
6. `06_filtering.py` - Filtreleme ve bulanıklaştırma
7. `07_edge_detection.py` - Kenar tespiti
8. `08_face_detection.py` - Yüz tespiti
9. `09_ocr.py` - Optik Karakter Tanıma (OCR) ve görüntü ön işleme

## 🖥️ Kullanım

Her script bağımsız olarak çalıştırılabilir:

```bash
python scripts/01_basics.py
```

## 📝 OCR Uygulaması

OCR uygulamasını kullanmak için Tesseract OCR'ın sisteminizde kurulu olması gerekmektedir:

- Windows: [Tesseract-OCR indirme sayfası](https://github.com/UB-Mannheim/tesseract/wiki)
- Linux: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`

Türkçe dil desteği için ilgili dil paketini de yüklemeniz gerekebilir.

OCR uygulaması, metin içeren görüntüleri işlemek için çeşitli ön işleme teknikleri sunar:

- Gri tonlama
- Eşikleme (basit, adaptif, Otsu)
- Bulanıklaştırma (Gaussian, iki taraflı)
- Morfolojik işlemler (genişletme, aşındırma, açma, kapama)

## 📋 Gereksinimler

- Python 3.6+
- OpenCV 4.5+
- NumPy
- Matplotlib
- Pillow
- PyTesseract
- Tkinter (GUI için)

## 👨‍🏫 Eğitmenler

- **Lokman Baturay Efe**
- **Niyazi Mert Işıksal**

## 📅 Etkinlik Detayları

- **Tarih:** 27 Şubat 2025
- **Saat:** 17:00
- **Yer:** Mühendislik Fakültesi D101
- **Organizasyon:** GDG On Campus Trakya University

---

_Not: Etkinliğe katılan ve katkıda bulunan herkese teşekkür ederiz!_
