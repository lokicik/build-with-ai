#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OpenCV Temel Kullanım Örneği
----------------------------
Bu script, OpenCV kütüphanesinin temel kullanımını göstermektedir:
- Görüntü okuma
- Görüntü gösterme
- Görüntü kaydetme
- Temel görüntü bilgilerini alma
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    print("OpenCV Temel Kullanım Örneği")
    print("-" * 30)
    
    # OpenCV sürümünü kontrol et
    print(f"OpenCV Sürümü: {cv2.__version__}")
    
    # Örnek bir görüntü oluştur (eğer images klasöründe görüntü yoksa)
    sample_img_path = "../images/sample.jpg"

    if not os.path.exists(sample_img_path):
        # Örnek bir görüntü oluştur
        img = np.zeros((300, 500, 3), dtype=np.uint8)
        # Mavi arka plan
        img[:, :] = (255, 0, 0)  # BGR formatında
        # Yeşil dikdörtgen
        cv2.rectangle(img, (100, 50), (400, 250), (0, 255, 0), -1)
        # Kırmızı metin
        cv2.putText(img, "OpenCV Egitimi", (120, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Görüntüyü kaydet
        os.makedirs("../images", exist_ok=True)
        cv2.imwrite(sample_img_path, img)
        print(f"Örnek görüntü oluşturuldu: {sample_img_path}")
    
    # 1. Görüntü Okuma
    print("\n1. Görüntü Okuma")
    img = cv2.imread(sample_img_path)
    
    if img is None:
        print(f"Hata: Görüntü okunamadı: {sample_img_path}")
        return
    
    # 2. Görüntü Bilgilerini Alma
    print("\n2. Görüntü Bilgileri")
    height, width, channels = img.shape
    print(f"Boyut: {width}x{height}")
    print(f"Kanal Sayısı: {channels}")
    print(f"Veri Tipi: {img.dtype}")
    print(f"Toplam Piksel Sayısı: {img.size}")
    
    # 3. Görüntü Gösterme (OpenCV ile)
    print("\n3. Görüntü Gösterme")
    cv2.imshow("OpenCV ile Görüntü", img)
    print("Görüntü penceresi açıldı. Devam etmek için herhangi bir tuşa basın...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 4. BGR'den RGB'ye Dönüştürme (Matplotlib için)
    print("\n4. BGR'den RGB'ye Dönüştürme")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 5. Matplotlib ile Görüntüleme
    print("\n5. Matplotlib ile Görüntüleme")
    plt.figure(figsize=(10, 6))
    
    plt.subplot(121)
    plt.title("BGR (OpenCV Formatı)")
    plt.imshow(img)  # BGR formatında
    
    plt.subplot(122)
    plt.title("RGB (Doğru Renkler)")
    plt.imshow(img_rgb)  # RGB formatında
    
    plt.tight_layout()
    plt.savefig("../images/color_comparison.png")
    print("Renk karşılaştırması kaydedildi: ../images/color_comparison.png")
    plt.show()
    
    # 6. Görüntü Kaydetme
    print("\n6. Görüntü Kaydetme")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("../images/sample_gray.jpg", gray_img)
    print("Gri tonlamalı görüntü kaydedildi: ../images/sample_gray.jpg")
    
    print("\nTemel OpenCV işlemleri tamamlandı!")

if __name__ == "__main__":
    main() 