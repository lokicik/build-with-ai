#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OpenCV Eşikleme Teknikleri
-------------------------
Bu script, OpenCV ile farklı eşikleme tekniklerini göstermektedir:
- Basit eşikleme
- Adaptif eşikleme
- Otsu eşikleme
- Renk tabanlı eşikleme (HSV)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def display_images(images, titles, filename=None, cmap='gray'):
    """Birden fazla görüntüyü yan yana gösterir ve kaydeder."""
    n = len(images)
    rows = (n + 2) // 3  # Her satırda en fazla 3 görüntü
    
    plt.figure(figsize=(15, 5 * rows))
    
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, min(n, 3), i+1)
        
        # Eğer görüntü BGR formatındaysa ve cmap 'gray' ise dönüştürme
        if cmap != 'gray' and len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
        print(f"Görüntü kaydedildi: {filename}")
    
    plt.show()

def create_gradient_image(width=500, height=100):
    """Soldan sağa doğru değişen bir gri tonlama gradyanı oluşturur."""
    gradient = np.zeros((height, width), dtype=np.uint8)
    for i in range(width):
        gradient[:, i] = int(255 * i / width)
    return gradient

def main():
    print("OpenCV Eşikleme Teknikleri")
    print("-" * 30)
    
    # 1. Gradyan görüntüsü üzerinde basit eşikleme
    print("\n1. Basit Eşikleme")
    
    # Gradyan görüntüsü oluştur
    gradient = create_gradient_image(500, 100)
    
    # Farklı eşikleme yöntemleri
    _, thresh_binary = cv2.threshold(gradient, 127, 255, cv2.THRESH_BINARY)
    _, thresh_binary_inv = cv2.threshold(gradient, 127, 255, cv2.THRESH_BINARY_INV)
    _, thresh_trunc = cv2.threshold(gradient, 127, 255, cv2.THRESH_TRUNC)
    _, thresh_tozero = cv2.threshold(gradient, 127, 255, cv2.THRESH_TOZERO)
    _, thresh_tozero_inv = cv2.threshold(gradient, 127, 255, cv2.THRESH_TOZERO_INV)
    
    # Sonuçları göster
    display_images(
        [gradient, thresh_binary, thresh_binary_inv, thresh_trunc, thresh_tozero, thresh_tozero_inv],
        ["Orijinal Gradyan", "İkili (Binary)", "İkili Ters (Binary Inv)", 
         "Kesme (Trunc)", "Sıfıra (ToZero)", "Sıfıra Ters (ToZero Inv)"],
        "../images/simple_thresholding.png"
    )
    
    # 2. Gerçek görüntü üzerinde basit eşikleme
    print("\n2. Gerçek Görüntü Üzerinde Basit Eşikleme")
    
    # Örnek görüntüyü yükle
    sample_img_path = "../images/sample.jpg"
    
    if not os.path.exists(sample_img_path):
        print(f"Hata: Örnek görüntü bulunamadı: {sample_img_path}")
        print("Lütfen önce 01_basics.py scriptini çalıştırın.")
        return
    
    # Görüntüyü oku ve gri tonlamaya dönüştür
    img = cv2.imread(sample_img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Farklı eşik değerleriyle ikili eşikleme
    _, thresh_50 = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
    _, thresh_100 = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
    _, thresh_150 = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
    _, thresh_200 = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    
    # Sonuçları göster
    display_images(
        [img_gray, thresh_50, thresh_100, thresh_150, thresh_200],
        ["Gri Tonlama", "Eşik = 50", "Eşik = 100", "Eşik = 150", "Eşik = 200"],
        "../images/real_image_thresholding.png"
    )
    
    # 3. Adaptif Eşikleme
    print("\n3. Adaptif Eşikleme")
    
    # Adaptif eşikleme yöntemleri
    # Ortalama adaptif eşikleme
    adaptive_mean = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Gaussian adaptif eşikleme
    adaptive_gaussian = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Farklı blok boyutlarıyla adaptif eşikleme
    adaptive_mean_5 = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2
    )
    
    adaptive_mean_21 = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2
    )
    
    # Sonuçları göster
    # display_images(
    #     [img_gray, adaptive_mean, adaptive_gaussian, adaptive_mean_5, adaptive_mean_21],
    #     ["Gri Tonlama", "Adaptif (Ortalama, 11x11)", "Adaptif (Gaussian, 11x11)",
    #      "Adaptif (Ortalama, 5x5)", "Adaptif (Ortalama, 21x21)"],
    #     "../images/adaptive_thresholding.png"
    # )
    
    # 4. Otsu Eşikleme
    print("\n4. Otsu Eşikleme")
    
    # Basit ikili eşikleme
    _, simple_thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    
    # Otsu eşikleme
    _, otsu_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_thresh_value = _
    print(f"Otsu algoritması tarafından hesaplanan eşik değeri: {otsu_thresh_value}")
    
    # Gürültü azaltma ve Otsu eşikleme
    # Önce Gaussian bulanıklaştırma uygula
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, otsu_blur_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_blur_thresh_value = _
    print(f"Bulanıklaştırma sonrası Otsu eşik değeri: {otsu_blur_thresh_value}")
    
    # Sonuçları göster
    display_images(
        [img_gray, simple_thresh, otsu_thresh, blur, otsu_blur_thresh],
        ["Gri Tonlama", "Basit Eşikleme (127)", f"Otsu Eşikleme ({otsu_thresh_value})", 
         "Gaussian Bulanıklaştırma", f"Bulanıklaştırma + Otsu ({otsu_blur_thresh_value})"],
        "../images/otsu_thresholding.png"
    )
    
    # 5. Renk Tabanlı Eşikleme (HSV)
    print("\n5. Renk Tabanlı Eşikleme (HSV)")
    
    # BGR'den HSV'ye dönüştür
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Mavi renk aralığı (HSV'de)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])
    
    # Yeşil renk aralığı (HSV'de)
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    
    # Kırmızı renk aralığı (HSV'de) - Kırmızı HSV'de iki aralığa sahiptir
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Renk maskeleri oluştur
    blue_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
    green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
    red_mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Maskeleri uygula
    blue_result = cv2.bitwise_and(img, img, mask=blue_mask)
    green_result = cv2.bitwise_and(img, img, mask=green_mask)
    red_result = cv2.bitwise_and(img, img, mask=red_mask)
    
    # Sonuçları göster
    display_images(
        [cv2.cvtColor(img, cv2.COLOR_BGR2RGB), blue_result, green_result, red_result],
        ["Orijinal", "Mavi Maske", "Yeşil Maske", "Kırmızı Maske"],
        "../images/color_thresholding.png",
        cmap=None
    )
    
    # 6. Çoklu Eşikleme
    print("\n6. Çoklu Eşikleme")
    
    # Çoklu seviye eşikleme
    # Gri tonlama görüntüsünü 4 seviyeye ayır
    _, thresh1 = cv2.threshold(img_gray, 64, 64, cv2.THRESH_BINARY)
    _, thresh2 = cv2.threshold(img_gray, 128, 128, cv2.THRESH_BINARY)
    _, thresh3 = cv2.threshold(img_gray, 192, 192, cv2.THRESH_BINARY)
    
    # Eşiklenmiş görüntüleri birleştir
    multi_thresh = cv2.bitwise_or(cv2.bitwise_or(thresh1, thresh2), thresh3)
    
    # Sonuçları göster
    display_images(
        [img_gray, thresh1, thresh2, thresh3, multi_thresh],
        ["Gri Tonlama", "Eşik = 64", "Eşik = 128", "Eşik = 192", "Çoklu Eşikleme"],
        "../images/multi_level_thresholding.png"
    )
    
    print("\nEşikleme teknikleri tamamlandı!")

if __name__ == "__main__":
    main() 