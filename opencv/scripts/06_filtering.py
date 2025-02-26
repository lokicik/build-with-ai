#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OpenCV Filtreleme ve Bulanıklaştırma
-----------------------------------
Bu script, OpenCV ile farklı filtreleme ve bulanıklaştırma tekniklerini göstermektedir:
- Ortalama (Mean) Bulanıklaştırma
- Gaussian Bulanıklaştırma
- Medyan Bulanıklaştırma
- İki Taraflı (Bilateral) Filtreleme
- Özel Filtreler (Keskinleştirme, Kabartma)
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

def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """Görüntüye tuz ve biber gürültüsü ekler."""
    noisy = np.copy(image)
    
    # Tuz gürültüsü (beyaz noktalar)
    salt_mask = np.random.random(image.shape[:2]) < salt_prob
    noisy[salt_mask] = 255
    
    # Biber gürültüsü (siyah noktalar)
    pepper_mask = np.random.random(image.shape[:2]) < pepper_prob
    noisy[pepper_mask] = 0
    
    return noisy

def add_gaussian_noise(image, mean=0, sigma=25):
    """Görüntüye Gaussian gürültüsü ekler."""
    # Görüntüyü float32'ye dönüştür
    img_float = image.astype(np.float32) / 255.0
    
    # Gaussian gürültüsü oluştur
    noise = np.random.normal(mean, sigma/255.0, image.shape)
    
    # Gürültüyü ekle ve sınırla
    noisy = img_float + noise
    noisy = np.clip(noisy, 0, 1)
    
    # Tekrar uint8'e dönüştür
    return (noisy * 255).astype(np.uint8)

def main():
    print("OpenCV Filtreleme ve Bulanıklaştırma")
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
    
    # Görüntüyü gri tonlamaya dönüştür
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Ortalama (Mean) Bulanıklaştırma
    print("\n1. Ortalama (Mean) Bulanıklaştırma")
    
    # Farklı çekirdek boyutlarıyla ortalama bulanıklaştırma
    blur_3x3 = cv2.blur(img, (3, 3))
    blur_5x5 = cv2.blur(img, (5, 5))
    blur_9x9 = cv2.blur(img, (9, 9))
    blur_15x15 = cv2.blur(img, (15, 15))
    
    # Sonuçları göster
    display_images(
        [img, blur_3x3, blur_5x5, blur_9x9, blur_15x15],
        ["Orijinal", "Ortalama 3x3", "Ortalama 5x5", "Ortalama 9x9", "Ortalama 15x15"],
        "../images/mean_blur.png"
    )
    
    # 2. Gaussian Bulanıklaştırma
    print("\n2. Gaussian Bulanıklaştırma")
    
    # Farklı çekirdek boyutlarıyla Gaussian bulanıklaştırma
    gaussian_3x3 = cv2.GaussianBlur(img, (3, 3), 0)
    gaussian_5x5 = cv2.GaussianBlur(img, (5, 5), 0)
    gaussian_9x9 = cv2.GaussianBlur(img, (9, 9), 0)
    gaussian_15x15 = cv2.GaussianBlur(img, (15, 15), 0)
    
    # # Sonuçları göster
    # display_images(
    #     [img, gaussian_3x3, gaussian_5x5, gaussian_9x9, gaussian_15x15],
    #     ["Orijinal", "Gaussian 3x3", "Gaussian 5x5", "Gaussian 9x9", "Gaussian 15x15"],
    #     "../images/gaussian_blur.png"
    # )
    
    # 3. Medyan Bulanıklaştırma
    print("\n3. Medyan Bulanıklaştırma")
    
    # Tuz ve biber gürültüsü ekle
    noisy_img = add_salt_pepper_noise(img_gray, 0.02, 0.02)
    
    # Farklı çekirdek boyutlarıyla medyan bulanıklaştırma
    median_3x3 = cv2.medianBlur(noisy_img, 3)
    median_5x5 = cv2.medianBlur(noisy_img, 5)
    median_7x7 = cv2.medianBlur(noisy_img, 7)
    
    # Karşılaştırma için Gaussian bulanıklaştırma
    gaussian_noisy = cv2.GaussianBlur(noisy_img, (5, 5), 0)
    
    # Sonuçları göster
    # display_images(
    #     [img_gray, noisy_img, gaussian_noisy, median_3x3, median_5x5, median_7x7],
    #     ["Orijinal", "Tuz ve Biber Gürültüsü", "Gaussian 5x5",
    #      "Medyan 3x3", "Medyan 5x5", "Medyan 7x7"],
    #     "../images/median_blur.png",
    #     cmap='gray'
    # )
    
    # 4. İki Taraflı (Bilateral) Filtreleme
    print("\n4. İki Taraflı (Bilateral) Filtreleme")
    
    # Gaussian gürültüsü ekle
    gaussian_noisy_img = add_gaussian_noise(img)
    
    # İki taraflı filtreleme
    bilateral_5 = cv2.bilateralFilter(img, 5, 75, 75)
    bilateral_9 = cv2.bilateralFilter(img, 9, 75, 75)
    bilateral_15 = cv2.bilateralFilter(img, 15, 75, 75)
    
    # Gaussian gürültülü görüntüye uygula
    bilateral_noisy = cv2.bilateralFilter(gaussian_noisy_img, 9, 75, 75)
    gaussian_noisy_blur = cv2.GaussianBlur(gaussian_noisy_img, (9, 9), 0)
    
    # Sonuçları göster
    # display_images(
    #     [img, bilateral_5, bilateral_9, bilateral_15],
    #     ["Orijinal", "İki Taraflı d=5", "İki Taraflı d=9", "İki Taraflı d=15"],
    #     "../images/bilateral_filter.png"
    # )
    
    display_images(
        [img, gaussian_noisy_img, gaussian_noisy_blur, bilateral_noisy],
        ["Orijinal", "Gaussian Gürültülü", "Gaussian Bulanıklaştırma", "İki Taraflı Filtreleme"],
        "../images/bilateral_vs_gaussian.png"
    )
    
    # 5. Özel Filtreler
    print("\n5. Özel Filtreler")
    
    # Keskinleştirme filtresi
    kernel_sharpen_1 = np.array([[-1, -1, -1],
                                 [-1,  9, -1],
                                 [-1, -1, -1]])
    
    kernel_sharpen_2 = np.array([[0, -1, 0],
                                 [-1, 5, -1],
                                 [0, -1, 0]])
    
    # Kabartma (Emboss) filtresi
    kernel_emboss = np.array([[-2, -1, 0],
                              [-1,  1, 1],
                              [ 0,  1, 2]])
    
    # Kenar algılama filtresi
    kernel_edge = np.array([[-1, -1, -1],
                            [-1,  8, -1],
                            [-1, -1, -1]])
    
    # Filtreleri uygula
    sharpen_1 = cv2.filter2D(img, -1, kernel_sharpen_1)
    sharpen_2 = cv2.filter2D(img, -1, kernel_sharpen_2)
    emboss = cv2.filter2D(img, -1, kernel_emboss)
    edge = cv2.filter2D(img, -1, kernel_edge)
    
    # Sonuçları göster
    display_images(
        [img, sharpen_1, sharpen_2, emboss, edge],
        ["Orijinal", "Keskinleştirme 1", "Keskinleştirme 2", "Kabartma", "Kenar Algılama"],
        "../images/custom_filters.png"
    )
    
    # 6. Bulanıklaştırma Karşılaştırması
    print("\n6. Bulanıklaştırma Karşılaştırması")
    
    # Tüm bulanıklaştırma yöntemlerini karşılaştır
    blur_mean = cv2.blur(img, (9, 9))
    blur_gaussian = cv2.GaussianBlur(img, (9, 9), 0)
    blur_median = cv2.medianBlur(img, 9)
    blur_bilateral = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Sonuçları göster
    display_images(
        [img, blur_mean, blur_gaussian, blur_median, blur_bilateral],
        ["Orijinal", "Ortalama 9x9", "Gaussian 9x9", "Medyan 9x9", "İki Taraflı d=9"],
        "../images/blur_comparison.png"
    )
    
    print("\nFiltreleme ve bulanıklaştırma işlemleri tamamlandı!")

if __name__ == "__main__":
    main() 