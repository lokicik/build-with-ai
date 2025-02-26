#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OpenCV Görüntü İşleme Operasyonları
-----------------------------------
Bu script, OpenCV ile temel görüntü işleme operasyonlarını göstermektedir:
- Yeniden boyutlandırma
- Kırpma
- Döndürme
- Aynalama
- Afin dönüşümleri
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def display_images(images, titles, filename=None):
    """Birden fazla görüntüyü yan yana gösterir ve kaydeder."""
    plt.figure(figsize=(15, 5))
    
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i+1)
        
        # Eğer görüntü BGR formatındaysa RGB'ye dönüştür
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
        print(f"Görüntü kaydedildi: {filename}")
    
    plt.show()

def main():
    print("OpenCV Görüntü İşleme Operasyonları")
    print("-" * 40)
    
    # Örnek görüntüyü yükle veya oluştur
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
    
    # Orijinal görüntü boyutları
    height, width = img.shape[:2]
    print(f"Orijinal görüntü boyutu: {width}x{height}")
    
    # 1. Yeniden Boyutlandırma
    print("\n1. Yeniden Boyutlandırma")
    
    # Belirli bir boyuta yeniden boyutlandırma
    resized_fixed = cv2.resize(img, (300, 200))
    print(f"Sabit boyuta yeniden boyutlandırma: 300x200")
    
    # Oranları koruyarak yeniden boyutlandırma
    scale_factor = 0.5
    resized_scaled = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
    print(f"Ölçekli yeniden boyutlandırma: {scale_factor} oranında")
    
    # Farklı interpolasyon yöntemleri
    resized_nearest = cv2.resize(img, (900, 600), interpolation=cv2.INTER_NEAREST)
    resized_cubic = cv2.resize(img, (900, 600), interpolation=cv2.INTER_CUBIC)
    
    # Sonuçları göster
    display_images(
        [img, resized_fixed, resized_scaled],
        ["Orijinal", "Sabit Boyut (300x200)", f"Ölçekli ({scale_factor})"],
        "../images/resizing.png"
    )
    
    # display_images(
    #     [resized_nearest, resized_cubic],
    #     ["En Yakın Komşu İnterpolasyonu", "Kübik İnterpolasyon"],
    #     "../images/interpolation.png"
    # )
    
    # 2. Kırpma
    print("\n2. Kırpma")
    
    # Görüntünün ortasından bir bölge kırp
    center_x, center_y = width // 2, height // 2
    crop_width, crop_height = 200, 150
    
    start_x = center_x - crop_width // 2
    start_y = center_y - crop_height // 2
    
    cropped = img[start_y:start_y+crop_height, start_x:start_x+crop_width]
    print(f"Kırpılan bölge: ({start_x},{start_y}) - ({start_x+crop_width},{start_y+crop_height})")
    
    # Sonuçları göster
    display_images(
        [img, cropped],
        ["Orijinal", "Kırpılmış Bölge"],
        "../images/cropping.png"
    )
    
    # 3. Döndürme
    print("\n3. Döndürme")
    
    # Görüntü merkezini hesapla
    center = (width // 2, height // 2)
    
    # 45 derece döndürme
    angle = 45
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_45 = cv2.warpAffine(img, rotation_matrix, (width, height))
    
    # 90 derece döndürme
    angle = 90
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_90 = cv2.warpAffine(img, rotation_matrix, (width, height))
    
    # Sonuçları göster
    display_images(
        [img, rotated_45, rotated_90],
        ["Orijinal", "45° Döndürülmüş", "90° Döndürülmüş"],
        "../images/rotation.png"
    )
    
    # 4. Aynalama
    print("\n4. Aynalama")
    
    # Yatay aynalama
    flipped_horizontal = cv2.flip(img, 1)  # 1 = yatay aynalama
    
    # Dikey aynalama
    flipped_vertical = cv2.flip(img, 0)    # 0 = dikey aynalama
    
    # Her iki yönde aynalama
    flipped_both = cv2.flip(img, -1)       # -1 = her iki yönde aynalama
    
    # Sonuçları göster
    display_images(
        [img, flipped_horizontal, flipped_vertical, flipped_both],
        ["Orijinal", "Yatay Aynalama", "Dikey Aynalama", "Her İki Yönde Aynalama"],
        "../images/flipping.png"
    )
    
    # 5. Afin Dönüşümü
    print("\n5. Afin Dönüşümü")
    
    # Afin dönüşümü için kaynak ve hedef noktaları tanımla
    rows, cols = img.shape[:2]
    
    # Kaynak noktaları (üçgen)
    src_points = np.float32([
        [0, 0],             # Sol üst
        [cols - 1, 0],      # Sağ üst
        [0, rows - 1]       # Sol alt
    ])
    
    # Hedef noktaları (üçgen)
    dst_points = np.float32([
        [cols * 0.2, rows * 0.1],  # Sol üst (sağa ve aşağı kaydırılmış)
        [cols * 0.8, rows * 0.2],  # Sağ üst (sola ve aşağı kaydırılmış)
        [cols * 0.1, rows * 0.9]   # Sol alt (sağa ve yukarı kaydırılmış)
    ])
    
    # Afin dönüşüm matrisini hesapla
    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
    
    # Afin dönüşümünü uygula
    affine_transformed = cv2.warpAffine(img, affine_matrix, (cols, rows))
    
    # Sonuçları göster
    display_images(
        [img, affine_transformed],
        ["Orijinal", "Afin Dönüşümü"],
        "../images/affine_transform.png"
    )
    
    print("\nGörüntü işleme operasyonları tamamlandı!")

if __name__ == "__main__":
    main() 