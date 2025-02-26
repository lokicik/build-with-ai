#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OpenCV ile Şekil Çizme ve Metin Ekleme
--------------------------------------
Bu script, OpenCV ile temel çizim işlemlerini göstermektedir:
- Çizgi çizme
- Dikdörtgen çizme
- Daire çizme
- Elips çizme
- Çokgen çizme
- Metin ekleme
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def display_image(img, title="Görüntü", filename=None):
    """Bir görüntüyü gösterir ve kaydeder."""
    # BGR'den RGB'ye dönüştür
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img
    
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis('off')
    
    if filename:
        plt.savefig(filename)
        print(f"Görüntü kaydedildi: {filename}")
    
    plt.show()

def main():
    print("OpenCV ile Sekil Cizme ve Metin Ekleme")
    print("-" * 40)
    
    # Boş bir tuval oluştur (siyah arka plan)
    canvas = np.zeros((500, 700, 3), dtype=np.uint8)
    
    # 1. Çizgi Çizme
    print("\n1. Cizgi Cizme")
    
    # Düz çizgi
    cv2.line(canvas, (50, 50), (650, 50), (0, 0, 255), 2)  # Kırmızı çizgi
    
    # Kesikli çizgi
    for x in range(50, 651, 20):
        cv2.line(canvas, (x, 80), (x+10, 80), (0, 255, 0), 2)  # Yeşil kesikli çizgi
    
    # Farklı kalınlıklarda çizgiler
    for i, thickness in enumerate([1, 2, 3, 4, 5]):
        y = 120 + i * 20
        cv2.line(canvas, (50, y), (650, y), (255, 255, 255), thickness)
        cv2.putText(canvas, f"Kalınlık: {thickness}", (660, y+5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 2. Dikdörtgen Çizme
    print("\n2. Dikdortgen Cizme")
    
    # Boş dikdörtgen
    cv2.rectangle(canvas, (50, 220), (200, 320), (255, 0, 0), 2)  # Mavi dikdörtgen
    
    # Dolu dikdörtgen
    cv2.rectangle(canvas, (250, 220), (400, 320), (0, 255, 255), -1)  # Sarı dolu dikdörtgen
    
    # Yuvarlak köşeli dikdörtgen
    cv2.rectangle(canvas, (450, 220), (600, 320), (255, 0, 255), 2, cv2.LINE_AA)  # Mor dikdörtgen
    
    # 3. Daire Çizme
    print("\n3. Daire Cizme")
    
    # Boş daire
    cv2.circle(canvas, (125, 400), 50, (0, 255, 0), 2)  # Yeşil daire
    
    # Dolu daire
    cv2.circle(canvas, (325, 400), 50, (255, 0, 0), -1)  # Mavi dolu daire
    
    # Yarım daire
    cv2.ellipse(canvas, (525, 400), (50, 50), 0, 0, 180, (0, 0, 255), -1)  # Kırmızı yarım daire
    
    # 4. Elips Çizme
    print("\n4. Elips Cizme")
    
    # Yeni bir tuval oluştur
    ellipse_canvas = np.zeros((500, 700, 3), dtype=np.uint8)
    
    # Boş elips
    cv2.ellipse(ellipse_canvas, (150, 150), (100, 50), 0, 0, 360, (0, 255, 255), 2)  # Sarı elips
    
    # Döndürülmüş elips
    cv2.ellipse(ellipse_canvas, (400, 150), (100, 50), 45, 0, 360, (255, 0, 255), 2)  # Mor döndürülmüş elips
    
    # Kısmi elips
    cv2.ellipse(ellipse_canvas, (150, 300), (100, 50), 0, 0, 180, (255, 255, 0), 2)  # Sarı yarım elips
    
    # Dolu elips
    cv2.ellipse(ellipse_canvas, (400, 300), (100, 50), 0, 0, 360, (0, 255, 255), -1)  # Dolu sarı elips
    
    # 5. Çokgen Çizme
    print("\n5. Cokgen Cizme")
    
    # Üçgen
    triangle_pts = np.array([[150, 400], [100, 480], [200, 480]], np.int32)
    triangle_pts = triangle_pts.reshape((-1, 1, 2))
    cv2.polylines(ellipse_canvas, [triangle_pts], True, (0, 255, 0), 2)  # Yeşil üçgen
    
    # Dolu beşgen
    pentagon_pts = np.array([[400, 400], [350, 430], [370, 480], [430, 480], [450, 430]], np.int32)
    pentagon_pts = pentagon_pts.reshape((-1, 1, 2))
    cv2.fillPoly(ellipse_canvas, [pentagon_pts], (0, 0, 255))  # Kırmızı dolu beşgen
    
    # Yıldız
    star_pts = np.array([
        [600, 380], [620, 420], [670, 420], [630, 450],
        [650, 500], [600, 470], [550, 500], [570, 450],
        [530, 420], [580, 420]
    ], np.int32)
    star_pts = star_pts.reshape((-1, 1, 2))
    cv2.polylines(ellipse_canvas, [star_pts], True, (255, 255, 255), 2)  # Beyaz yıldız
    
    # 6. Metin Ekleme
    print("\n6. Metin Ekleme")
    
    # Yeni bir tuval oluştur
    text_canvas = np.zeros((500, 700, 3), dtype=np.uint8)
    
    # Farklı fontlar
    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX,
             cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL,
             cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX]
    
    font_names = ["SIMPLEX", "PLAIN", "DUPLEX", "COMPLEX", "TRIPLEX", 
                 "COMPLEX_SMALL", "SCRIPT_SIMPLEX", "SCRIPT_COMPLEX"]
    
    # Farklı fontları göster
    for i, (font, name) in enumerate(zip(fonts, font_names)):
        y = 50 + i * 50
        cv2.putText(text_canvas, f"OpenCV Font: {name}", (50, y), 
                   font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Farklı kalınlıklarda metin
    for i, thickness in enumerate([1, 2, 3, 4]):
        y = 50 + i * 50
        cv2.putText(text_canvas, f"Kalinlik: {thickness}", (400, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness, cv2.LINE_AA)
    
    # Farklı boyutlarda metin
    for i, scale in enumerate([0.5, 1.0, 1.5, 2.0]):
        y = 250 + i * 50
        cv2.putText(text_canvas, f"Olcek: {scale}", (50, y),
                   cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 255), 2, cv2.LINE_AA)
    
    # Farklı renklerde metin
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    color_names = ["Mavi", "Yesil", "Kirmizi", "Sari", "Camgobegi", "Mor"]
    
    for i, (color, name) in enumerate(zip(colors, color_names)):
        y = 250 + i * 40
        cv2.putText(text_canvas, f"Renk: {name}", (400, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    # Görüntüleri göster ve kaydet
    display_image(canvas, "Cizgiler ve Temel Sekiller", "../images/basic_shapes.png")
    display_image(ellipse_canvas, "Elipsler ve Cokgenler", "../images/ellipses_polygons.png")
    display_image(text_canvas, "Metin Ornekleri", "../images/text_examples.png")
    
    # 7. Tüm çizimleri birleştir
    print("\n7. Tüm Cizimleri Birlestirme")
    
    # Yeni bir tuval oluştur
    final_canvas = np.zeros((600, 800, 3), dtype=np.uint8)
    
    # Arka plan rengi
    final_canvas[:] = (50, 50, 50)  # Koyu gri arka plan
    
    # Dikdörtgen çiz
    cv2.rectangle(final_canvas, (50, 50), (750, 550), (100, 100, 100), 2)
    
    # Başlık ekle
    cv2.putText(final_canvas, "OpenCV ile Cizim Ornekleri", (150, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Şekiller çiz
    # Daire
    cv2.circle(final_canvas, (200, 150), 60, (0, 0, 255), -1)
    cv2.circle(final_canvas, (200, 150), 60, (255, 255, 255), 2)
    
    # Dikdörtgen
    cv2.rectangle(final_canvas, (350, 100), (500, 200), (0, 255, 0), -1)
    cv2.rectangle(final_canvas, (350, 100), (500, 200), (255, 255, 255), 2)
    
    # Üçgen
    triangle_pts = np.array([[600, 100], [550, 200], [650, 200]], np.int32)
    triangle_pts = triangle_pts.reshape((-1, 1, 2))
    cv2.fillPoly(final_canvas, [triangle_pts], (255, 0, 0))
    cv2.polylines(final_canvas, [triangle_pts], True, (255, 255, 255), 2)
    
    # Elips
    cv2.ellipse(final_canvas, (200, 350), (100, 50), 0, 0, 360, (255, 255, 0), -1)
    cv2.ellipse(final_canvas, (200, 350), (100, 50), 0, 0, 360, (255, 255, 255), 2)
    
    # Beşgen
    pentagon_pts = np.array([[400, 300], [350, 350], [370, 400], [430, 400], [450, 350]], np.int32)
    pentagon_pts = pentagon_pts.reshape((-1, 1, 2))
    cv2.fillPoly(final_canvas, [pentagon_pts], (0, 255, 255))
    cv2.polylines(final_canvas, [pentagon_pts], True, (255, 255, 255), 2)
    
    # Yıldız
    star_pts = np.array([
        [600, 300], [620, 340], [670, 340], [630, 370],
        [650, 420], [600, 390], [550, 420], [570, 370],
        [530, 340], [580, 340]
    ], np.int32)
    star_pts = star_pts.reshape((-1, 1, 2))
    cv2.fillPoly(final_canvas, [star_pts], (255, 0, 255))
    cv2.polylines(final_canvas, [star_pts], True, (255, 255, 255), 2)
    
    # Metin ekle
    cv2.putText(final_canvas, "Daire", (180, 150), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.putText(final_canvas, "Dikdortgen", (380, 150),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.putText(final_canvas, "Ucgen", (580, 150),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.putText(final_canvas, "Elips", (180, 350), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.putText(final_canvas, "Besgen", (380, 350),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.putText(final_canvas, "Yildiz", (580, 350),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Çizgi çiz
    cv2.line(final_canvas, (50, 450), (750, 450), (255, 255, 255), 2)
    
    # Alt başlık ekle
    cv2.putText(final_canvas, "OpenCV ile goruntu isleme ve bilgisayarli goru", (150, 500),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Görüntüyü göster ve kaydet
    display_image(final_canvas, "Tüm Cizimler", "../images/all_drawings.png")
    
    print("\nŞekil çizme ve metin ekleme işlemleri tamamlandı!")

if __name__ == "__main__":
    main() 