#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OpenCV Renk Uzayları ve Dönüşümleri
-----------------------------------
Bu script, OpenCV ile farklı renk uzaylarını ve aralarındaki dönüşümleri göstermektedir:
- RGB/BGR
- Gri tonlama
- HSV (Hue, Saturation, Value)
- LAB
- YCrCb
- Renk kanallarına ayırma ve birleştirme
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def display_images(images, titles, filename=None, cmap=None):
    """Birden fazla görüntüyü yan yana gösterir ve kaydeder."""
    n = len(images)
    rows = (n + 2) // 3  # Her satırda en fazla 3 görüntü
    
    plt.figure(figsize=(15, 5 * rows))
    
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, min(n, 3), i+1)
        
        # Eğer görüntü BGR formatındaysa ve cmap belirtilmemişse RGB'ye dönüştür
        if cmap is None and len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
        print(f"Görüntü kaydedildi: {filename}")
    
    plt.show()

def display_channels(channels, channel_names, color_maps, filename=None):
    """Renk kanallarını ayrı ayrı gösterir."""
    n = len(channels)
    plt.figure(figsize=(15, 5))
    
    for i, (channel, name, cmap) in enumerate(zip(channels, channel_names, color_maps)):
        plt.subplot(1, n, i+1)
        plt.imshow(channel, cmap=cmap)
        plt.title(name)
        plt.axis('off')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
        print(f"Görüntü kaydedildi: {filename}")
    
    plt.show()

def main():
    print("OpenCV Renk Uzayları ve Dönüşümleri")
    print("-" * 40)
    
    # Örnek görüntüyü yükle
    sample_img_path = "../images/sample.jpg"
    
    if not os.path.exists(sample_img_path):
        print(f"Hata: Örnek görüntü bulunamadı: {sample_img_path}")
        print("Lütfen önce 01_basics.py scriptini çalıştırın.")
        return
    
    # Görüntüyü oku
    img = cv2.imread(sample_img_path)
    
    if img is None:
        print(f"Hata: Görüntü okunamadı: {sample_img_path}")
        return
    
    # 1. BGR ve RGB
    print("\n1. BGR ve RGB Renk Uzayları")
    
    # OpenCV BGR formatından RGB'ye dönüştürme
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Sonuçları göster
    display_images(
        [img, img_rgb],
        ["BGR (OpenCV Formatı)", "RGB"],
        "../images/bgr_rgb.png"
    )
    
    # 2. Gri Tonlama
    print("\n2. Gri Tonlama")
    
    # BGR'den gri tonlamaya dönüştürme
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Sonuçları göster
    display_images(
        [img_rgb, img_gray],
        ["Orijinal (RGB)", "Gri Tonlama"],
        "../images/grayscale.png",
        cmap='gray'
    )
    
    # 3. HSV (Hue, Saturation, Value)
    print("\n3. HSV Renk Uzayı")
    
    # BGR'den HSV'ye dönüştürme
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # HSV kanallarını ayır
    h, s, v = cv2.split(img_hsv)
    
    # Sonuçları göster
    display_images(
        [img_rgb, img_hsv],
        ["Orijinal (RGB)", "HSV"],
        "../images/hsv.png"
    )
    
    # HSV kanallarını göster
    display_channels(
        [h, s, v],
        ["Hue (Renk Tonu)", "Saturation (Doygunluk)", "Value (Parlaklık)"],
        ['hsv', 'gray', 'gray'],
        "../images/hsv_channels.png"
    )
    
    # 4. LAB Renk Uzayı
    print("\n4. LAB Renk Uzayı")
    
    # BGR'den LAB'a dönüştürme
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # LAB kanallarını ayır
    l, a, b = cv2.split(img_lab)
    
    # # Sonuçları göster
    # display_images(
    #     [img_rgb, img_lab],
    #     ["Orijinal (RGB)", "LAB"],
    #     "../images/lab.png"
    # )
    #
    # # LAB kanallarını göster
    # display_channels(
    #     [l, a, b],
    #     ["L (Lightness)", "A (Green-Red)", "B (Blue-Yellow)"],
    #     ['gray', 'gray', 'gray'],
    #     "../images/lab_channels.png"
    # )
    
    # 5. YCrCb Renk Uzayı
    print("\n5. YCrCb Renk Uzayı")
    
    # BGR'den YCrCb'ye dönüştürme
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    # YCrCb kanallarını ayır
    y, cr, cb = cv2.split(img_ycrcb)
    
    # # Sonuçları göster
    # display_images(
    #     [img_rgb, img_ycrcb],
    #     ["Orijinal (RGB)", "YCrCb"],
    #     "../images/ycrcb.png"
    # )
    #
    # # YCrCb kanallarını göster
    # display_channels(
    #     [y, cr, cb],
    #     ["Y (Luminance)", "Cr (Red-Difference)", "Cb (Blue-Difference)"],
    #     ['gray', 'gray', 'gray'],
    #     "../images/ycrcb_channels.png"
    # )
    
    # 6. BGR Kanallarına Ayırma ve Birleştirme
    print("\n6. BGR Kanallarına Ayırma ve Birleştirme")
    
    # BGR kanallarını ayır
    b, g, r = cv2.split(img)
    
    # Sıfır matrisleri oluştur
    zeros = np.zeros_like(b)
    
    # Sadece bir kanalı içeren görüntüler oluştur
    img_b = cv2.merge([b, zeros, zeros])  # Sadece mavi kanal
    img_g = cv2.merge([zeros, g, zeros])  # Sadece yeşil kanal
    img_r = cv2.merge([zeros, zeros, r])  # Sadece kırmızı kanal
    
    # Kanalları birleştir
    img_merged = cv2.merge([b, g, r])
    
    # # BGR kanallarını göster
    display_channels(
        [b, g, r],
        ["Blue Channel", "Green Channel", "Red Channel"],
        ['Blues', 'Greens', 'Reds'],
        "../images/bgr_channels.png"
    )
    
    # Tek kanallı görüntüleri göster
    display_images(
        [img_rgb, img_b, img_g, img_r, img_merged],
        ["Orijinal", "Sadece Mavi", "Sadece Yeşil", "Sadece Kırmızı", "Birleştirilmiş"],
        "../images/bgr_isolated.png"
    )
    
    print("\nRenk uzayları ve dönüşümleri tamamlandı!")

if __name__ == "__main__":
    main() 