# OpenCV Görüntü İşleme Eğitimi

Bu repo, OpenCV kütüphanesi kullanarak temel görüntü işleme tekniklerini öğrenmek için hazırlanmış örnek scriptleri içermektedir.

## Kurulum

Gerekli kütüphaneleri yüklemek için:

```bash
pip install -r requirements.txt
```

## İçerik

1. `01_basics.py` - OpenCV'ye giriş ve temel işlemler
2. `02_image_operations.py` - Görüntü işleme operasyonları (yeniden boyutlandırma, kırpma, döndürme)
3. `03_color_spaces.py` - Renk uzayları ve dönüşümleri
4. `04_drawing.py` - Şekil çizme ve metin ekleme
5. `05_thresholding.py` - Eşikleme teknikleri
6. `06_filtering.py` - Filtreleme ve bulanıklaştırma
7. `07_edge_detection.py` - Kenar tespiti
8. `08_contours.py` - Kontur tespiti ve analizi
9. `09_face_detection.py` - Yüz tespiti
10. `10_ocr.py` - Optik Karakter Tanıma (OCR) ve görüntü ön işleme

## Kullanım

Her script bağımsız olarak çalıştırılabilir:

```bash
python scripts/01_basics.py
```

## OCR Uygulaması

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

## Gereksinimler

- Python 3.6+
- OpenCV 4.5+
- NumPy
- Matplotlib
- Pillow
- PyTesseract
- Tkinter (GUI için)
