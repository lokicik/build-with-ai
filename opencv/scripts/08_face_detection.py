#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OpenCV Yüz Tespiti
----------------
Bu script, OpenCV ile yüz tespiti işlemlerini göstermektedir:
- Haar Cascade sınıflandırıcıları ile yüz tespiti
- Göz tespiti
- Gülümseme tespiti
- Yüz işaretleri (landmarks) tespiti
- Gerçek zamanlı yüz tespiti
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request
import time

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

def download_cascade_files():
    """Haar Cascade XML dosyalarını indirir."""
    cascade_dir = "../cascades"
    os.makedirs(cascade_dir, exist_ok=True)
    
    # İndirilecek Haar Cascade dosyaları
    cascade_files = {
        "haarcascade_frontalface_default.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
        "haarcascade_eye.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml",
        "haarcascade_smile.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_smile.xml",
        "haarcascade_frontalface_alt.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml",
        "haarcascade_profileface.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_profileface.xml"
    }
    
    for filename, url in cascade_files.items():
        file_path = os.path.join(cascade_dir, filename)
        
        if not os.path.exists(file_path):
            print(f"İndiriliyor: {filename}")
            try:
                urllib.request.urlretrieve(url, file_path)
                print(f"İndirildi: {file_path}")
            except Exception as e:
                print(f"Hata: {filename} indirilemedi - {e}")
        else:
            print(f"Zaten mevcut: {file_path}")
    
    return cascade_dir

def download_test_image():
    """Test için örnek bir yüz görüntüsü indirir."""
    image_dir = "../images"
    os.makedirs(image_dir, exist_ok=True)
    
    # Örnek yüz görüntüsü
    face_image_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"
    face_image_path = os.path.join(image_dir, "face_sample.jpg")
    
    if not os.path.exists(face_image_path):
        print(f"Örnek yüz görüntüsü indiriliyor...")
        try:
            urllib.request.urlretrieve(face_image_url, face_image_path)
            print(f"İndirildi: {face_image_path}")
        except Exception as e:
            print(f"Hata: Örnek yüz görüntüsü indirilemedi - {e}")
    else:
        print(f"Örnek yüz görüntüsü zaten mevcut: {face_image_path}")
    
    # Grup yüz görüntüsü
    group_image_url = "../images/group_image.jpg"
    group_image_path = os.path.join(image_dir, "group_image.jpg")
    
    return face_image_path, group_image_path

def detect_faces(image, face_cascade):
    """Görüntüdeki yüzleri tespit eder."""
    # Gri tonlamaya dönüştür
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Yüzleri tespit et
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    return faces, gray

def detect_eyes(gray, face_roi, eye_cascade):
    """Yüz bölgesindeki gözleri tespit eder."""
    eyes = eye_cascade.detectMultiScale(
        face_roi,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    return eyes

def detect_smile(gray, face_roi, smile_cascade):
    """Yüz bölgesindeki gülümsemeleri tespit eder."""
    smiles = smile_cascade.detectMultiScale(
        face_roi,
        scaleFactor=1.8,
        minNeighbors=20,
        minSize=(25, 25),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    return smiles

def main():
    print("OpenCV Yüz Tespiti")
    print("-" * 20)
    
    # Haar Cascade dosyalarını indir
    cascade_dir = download_cascade_files()
    
    # Cascade sınıflandırıcılarını yükle
    face_cascade = cv2.CascadeClassifier(os.path.join(cascade_dir, 'haarcascade_frontalface_default.xml'))
    eye_cascade = cv2.CascadeClassifier(os.path.join(cascade_dir, 'haarcascade_eye.xml'))
    smile_cascade = cv2.CascadeClassifier(os.path.join(cascade_dir, 'haarcascade_smile.xml'))
    profile_cascade = cv2.CascadeClassifier(os.path.join(cascade_dir, 'haarcascade_profileface.xml'))
    
    # Test görüntülerini indir
    face_image_path, group_image_path = download_test_image()
    
    # 1. Temel Yüz Tespiti
    print("\n1. Temel Yüz Tespiti")
    
    # Görüntüyü oku
    img = cv2.imread(face_image_path)
    
    if img is None:
        print(f"Hata: Görüntü okunamadı: {face_image_path}")
        return
    
    # Yüzleri tespit et
    faces, gray = detect_faces(img, face_cascade)
    
    print(f"Tespit edilen yüz sayısı: {len(faces)}")
    
    # Yüzleri işaretle
    face_detection_img = img.copy()
    
    for (x, y, w, h) in faces:
        cv2.rectangle(face_detection_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Sonuçları göster
    display_images(
        [img, face_detection_img],
        ["Orijinal", f"Yüz Tespiti ({len(faces)} yüz)"],
        "../images/face_detection.png",
        cmap=None
    )
    
    # 2. Göz ve Gülümseme Tespiti
    print("\n2. Göz ve Gülümseme Tespiti")
    
    # Yüz, göz ve gülümseme tespiti
    eye_smile_img = img.copy()
    
    for (x, y, w, h) in faces:
        # Yüzü işaretle
        cv2.rectangle(eye_smile_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Yüz bölgesini al
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = eye_smile_img[y:y+h, x:x+w]
        
        # Gözleri tespit et
        eyes = detect_eyes(gray, roi_gray, eye_cascade)
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
        
        # Gülümsemeleri tespit et
        smiles = detect_smile(gray, roi_gray, smile_cascade)
        
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
    
    # Sonuçları göster
    display_images(
        [img, eye_smile_img],
        ["Orijinal", "Yüz, Göz ve Gülümseme Tespiti"],
        "../images/eye_smile_detection.png",
        cmap=None
    )
    
    # 3. Grup Fotoğrafında Yüz Tespiti
    print("\n3. Grup Fotoğrafında Yüz Tespiti")
    
    # Grup görüntüsünü oku
    group_img = cv2.imread(group_image_path)
    
    if group_img is None:
        print(f"Hata: Görüntü okunamadı: {group_image_path}")
    else:
        # Yüzleri tespit et
        group_faces, group_gray = detect_faces(group_img, face_cascade)
        
        print(f"Grup fotoğrafında tespit edilen yüz sayısı: {len(group_faces)}")
        
        # Yüzleri işaretle
        group_detection_img = group_img.copy()
        
        for i, (x, y, w, h) in enumerate(group_faces):
            cv2.rectangle(group_detection_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(group_detection_img, f"#{i+1}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Sonuçları göster
        display_images(
            [group_img, group_detection_img],
            ["Orijinal Grup", f"Yüz Tespiti ({len(group_faces)} yüz)"],
            "../images/group_face_detection.png",
            cmap=None
        )
    
    # 4. Farklı Parametrelerle Yüz Tespiti
    print("\n4. Farklı Parametrelerle Yüz Tespiti")
    
    # Farklı scaleFactor değerleriyle yüz tespiti
    scale_1_1_img = group_img.copy()
    scale_1_2_img = group_img.copy()
    scale_1_3_img = group_img.copy()
    
    # scaleFactor = 1.1
    faces_1_1 = face_cascade.detectMultiScale(
        group_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    
    # scaleFactor = 1.2
    faces_1_2 = face_cascade.detectMultiScale(
        group_gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30)
    )
    
    # scaleFactor = 1.3
    faces_1_3 = face_cascade.detectMultiScale(
        group_gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
    )
    
    # Yüzleri işaretle
    for (x, y, w, h) in faces_1_1:
        cv2.rectangle(scale_1_1_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    for (x, y, w, h) in faces_1_2:
        cv2.rectangle(scale_1_2_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    for (x, y, w, h) in faces_1_3:
        cv2.rectangle(scale_1_3_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Sonuçları göster
    display_images(
        [scale_1_1_img, scale_1_2_img, scale_1_3_img],
        [f"scaleFactor=1.1 ({len(faces_1_1)} yüz)", 
         f"scaleFactor=1.2 ({len(faces_1_2)} yüz)", 
         f"scaleFactor=1.3 ({len(faces_1_3)} yüz)"],
        "../images/face_detection_scale_factor.png",
        cmap=None
    )
    
    # Farklı minNeighbors değerleriyle yüz tespiti
    neighbors_3_img = group_img.copy()
    neighbors_5_img = group_img.copy()
    neighbors_7_img = group_img.copy()
    
    # minNeighbors = 3
    faces_n3 = face_cascade.detectMultiScale(
        group_gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
    )
    
    # minNeighbors = 5
    faces_n5 = face_cascade.detectMultiScale(
        group_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    
    # minNeighbors = 7
    faces_n7 = face_cascade.detectMultiScale(
        group_gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30)
    )
    
    # Yüzleri işaretle
    for (x, y, w, h) in faces_n3:
        cv2.rectangle(neighbors_3_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    for (x, y, w, h) in faces_n5:
        cv2.rectangle(neighbors_5_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    for (x, y, w, h) in faces_n7:
        cv2.rectangle(neighbors_7_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Sonuçları göster
    display_images(
        [neighbors_3_img, neighbors_5_img, neighbors_7_img],
        [f"minNeighbors=3 ({len(faces_n3)} yüz)", 
         f"minNeighbors=5 ({len(faces_n5)} yüz)", 
         f"minNeighbors=7 ({len(faces_n7)} yüz)"],
        "../images/face_detection_min_neighbors.png",
        cmap=None
    )
    
    # 5. Profil Yüz Tespiti
    print("\n5. Profil Yüz Tespiti")
    
    # Profil yüzleri tespit et
    profile_faces = profile_cascade.detectMultiScale(
        group_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    
    print(f"Tespit edilen profil yüz sayısı: {len(profile_faces)}")
    
    # Profil yüzleri işaretle
    profile_img = group_img.copy()
    
    for (x, y, w, h) in profile_faces:
        cv2.rectangle(profile_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Hem ön hem profil yüzleri işaretle
    combined_img = group_img.copy()
    
    for (x, y, w, h) in group_faces:
        cv2.rectangle(combined_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    for (x, y, w, h) in profile_faces:
        cv2.rectangle(combined_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Sonuçları göster
    display_images(
        [group_img, profile_img, combined_img],
        ["Orijinal", f"Profil Yüz Tespiti ({len(profile_faces)} yüz)", 
         "Ön ve Profil Yüz Tespiti"],
        "../images/profile_face_detection.png",
        cmap=None
    )
    
    # 6. Gerçek Zamanlı Yüz Tespiti (Kamera)
    print("\n6. Gerçek Zamanlı Yüz Tespiti")
    print("Kamera açılıyor... (Çıkmak için 'q' tuşuna basın)")
    
    # Kamera bağlantısını aç
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Hata: Kamera açılamadı!")
        return
    
    # FPS hesaplama için değişkenler
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    
    try:
        while True:
            # Kameradan bir kare oku
            ret, frame = cap.read()
            
            if not ret:
                print("Kameradan kare okunamadı!")
                break
            
            # FPS hesapla
            fps_frame_count += 1
            if (time.time() - fps_start_time) > 1:
                fps = fps_frame_count
                fps_frame_count = 0
                fps_start_time = time.time()
            
            # Yüzleri tespit et
            faces, gray = detect_faces(frame, face_cascade)
            
            # Yüzleri işaretle
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Yüz bölgesini al
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                
                # Gözleri tespit et
                eyes = detect_eyes(gray, roi_gray, eye_cascade)
                
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
                
                # Gülümsemeleri tespit et
                smiles = detect_smile(gray, roi_gray, smile_cascade)
                
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
            
            # FPS ve tespit bilgilerini ekle
            cv2.putText(frame, f"FPS: {fps}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Yüzler: {len(faces)}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Sonucu göster
            cv2.imshow('Gerçek Zamanlı Yüz Tespiti', frame)
            
            # 'q' tuşuna basılırsa çık
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # Kamera bağlantısını kapat
        cap.release()
        cv2.destroyAllWindows()
    
    print("\nYüz tespiti işlemleri tamamlandı!")

if __name__ == "__main__":
    main() 