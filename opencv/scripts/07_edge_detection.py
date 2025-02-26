#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OpenCV Kenar Tespiti
------------------
Bu script, OpenCV ile farklı kenar tespit yöntemlerini göstermektedir:
- Sobel Operatörü
- Scharr Operatörü
- Laplacian Operatörü
- Canny Kenar Dedektörü
- Gradyan Büyüklüğü ve Yönü
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

def main():
    print("OpenCV Kenar Tespiti")
    print("-" * 25)
    
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
    
    # Gürültüyü azaltmak için bulanıklaştırma uygula
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
    
    # 1. Sobel Operatörü
    print("\n1. Sobel Operatörü")
    
    # X yönünde Sobel
    sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = cv2.convertScaleAbs(sobel_x)  # Mutlak değer ve ölçekleme
    
    # Y yönünde Sobel
    sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    
    # X ve Y yönlerini birleştir
    sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    
    # Sonuçları göster
    # display_images(
    #     [img_gray, sobel_x, sobel_y, sobel_combined],
    #     ["Gri Tonlama", "Sobel X", "Sobel Y", "Sobel Birleşik"],
    #     "../images/sobel_edges.png"
    # )
    
    # 2. Scharr Operatörü
    print("\n2. Scharr Operatörü")
    
    # X yönünde Scharr
    scharr_x = cv2.Scharr(img_blur, cv2.CV_64F, 1, 0)
    scharr_x = cv2.convertScaleAbs(scharr_x)
    
    # Y yönünde Scharr
    scharr_y = cv2.Scharr(img_blur, cv2.CV_64F, 0, 1)
    scharr_y = cv2.convertScaleAbs(scharr_y)
    
    # X ve Y yönlerini birleştir
    scharr_combined = cv2.addWeighted(scharr_x, 0.5, scharr_y, 0.5, 0)
    
    # Sonuçları göster
    # display_images(
    #     [img_gray, scharr_x, scharr_y, scharr_combined],
    #     ["Gri Tonlama", "Scharr X", "Scharr Y", "Scharr Birleşik"],
    #     "../images/scharr_edges.png"
    # )
    
    # 3. Laplacian Operatörü
    print("\n3. Laplacian Operatörü")
    
    # Laplacian uygula
    laplacian = cv2.Laplacian(img_blur, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    
    # # Sonuçları göster
    # display_images(
    #     [img_gray, img_blur, laplacian],
    #     ["Gri Tonlama", "Bulanıklaştırılmış", "Laplacian"],
    #     "../images/laplacian_edges.png"
    # )
    
    # 4. Canny Kenar Dedektörü
    print("\n4. Canny Kenar Dedektörü")
    
    # Farklı eşik değerleriyle Canny
    canny_50_150 = cv2.Canny(img_blur, 50, 150)
    canny_10_150 = cv2.Canny(img_blur, 10, 150)
    canny_50_200 = cv2.Canny(img_blur, 50, 200)
    canny_100_200 = cv2.Canny(img_blur, 100, 200)
    
    # Sonuçları göster
    display_images(
        [img_gray, canny_50_150, canny_10_150, canny_50_200, canny_100_200],
        ["Gri Tonlama", "Canny (50, 150)", "Canny (10, 150)", 
         "Canny (50, 200)", "Canny (100, 200)"],
        "../images/canny_edges.png"
    )
    
    # 5. Gradyan Büyüklüğü ve Yönü
    print("\n5. Gradyan Büyüklüğü ve Yönü")
    
    # Sobel gradyanları hesapla (float olarak)
    sobelx64f = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely64f = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=3)
    
    # Gradyan büyüklüğü
    mag = cv2.magnitude(sobelx64f, sobely64f)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Gradyan yönü (radyan cinsinden)
    theta = np.arctan2(sobely64f, sobelx64f)
    
    # Yönü görselleştir (HSV renk uzayı kullanarak)
    h = (theta * 180 / np.pi) % 180  # Hue (0-180)
    s = np.ones_like(h) * 255        # Saturation (tam doygunluk)
    v = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value (büyüklük)
    
    # HSV görüntüsü oluştur
    hsv = np.stack([h, s, v], axis=2).astype(np.uint8)
    gradient_direction = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Sonuçları göster
    # display_images(
    #     [img_gray, mag],
    #     ["Gri Tonlama", "Gradyan Büyüklüğü"],
    #     "../images/gradient_magnitude.png"
    # )
    
    display_images(
        [img, gradient_direction],
        ["Orijinal", "Gradyan Yönü (Renkli)"],
        "../images/gradient_direction.png",
        cmap=None
    )
    
    # 6. Kenar Tespit Yöntemlerinin Karşılaştırması
    print("\n6. Kenar Tespit Yöntemlerinin Karşılaştırması")
    
    # Tüm yöntemleri karşılaştır
    display_images(
        [img_gray, sobel_combined, scharr_combined, laplacian, canny_50_150, mag],
        ["Orijinal", "Sobel", "Scharr", "Laplacian", "Canny", "Gradyan Büyüklüğü"],
        "../images/edge_detection_comparison.png"
    )
    
    # 7. Kenar Tespiti Uygulaması: Kenarları Renkli Görüntüde Vurgulama
    print("\n7. Kenar Tespiti Uygulaması: Kenarları Renkli Görüntüde Vurgulama")
    
    # Canny kenar tespiti
    edges = cv2.Canny(img_blur, 50, 150)
    
    # Kenarları BGR formatına dönüştür
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Kenarları kırmızı yap
    edges_red = edges_bgr.copy()
    edges_red[edges > 0] = [0, 0, 255]  # Kenarları kırmızı yap
    
    # Orijinal görüntü ile kenarları birleştir
    edges_overlay = cv2.addWeighted(img, 0.8, edges_red, 0.8, 0)
    
    # Sonuçları göster
    display_images(
        [img, edges, edges_red, edges_overlay],
        ["Orijinal", "Canny Kenarları", "Kırmızı Kenarlar", "Kenar Vurgulamalı"],
        "../images/edge_highlighting.png",
        cmap=None
    )
    
    print("\nKenar tespiti işlemleri tamamlandı!")

if __name__ == "__main__":
    main() 