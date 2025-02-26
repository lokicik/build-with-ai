#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OpenCV OCR (Optical Character Recognition) Uygulaması
----------------------------------------------------
Bu script, OpenCV ve Tesseract kullanarak OCR işlemini göstermektedir:
- Görüntü ön işleme adımları
- Tesseract OCR entegrasyonu
- Metin tanıma ve çıkarma
- Kullanıcı dostu GUI arayüzü
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pytesseract
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import threading
import time

# Tesseract yolunu ayarla (Windows için)
if sys.platform.startswith('win'):
    # Windows'ta Tesseract'ın kurulu olduğu yolu belirtin
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenCV OCR Uygulaması")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Ana çerçeve
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Sol panel (görüntü ve kontroller)
        self.left_panel = ttk.Frame(self.main_frame)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Sağ panel (sonuçlar)
        self.right_panel = ttk.Frame(self.main_frame)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Görüntü çerçevesi
        self.image_frame = ttk.LabelFrame(self.left_panel, text="Görüntü")
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Kontrol çerçevesi
        self.control_frame = ttk.LabelFrame(self.left_panel, text="Kontroller")
        self.control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Dosya seçme butonu
        self.file_button = ttk.Button(self.control_frame, text="Görüntü Seç", command=self.load_image)
        self.file_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        # Ön işleme seçenekleri
        self.preprocess_label = ttk.Label(self.control_frame, text="Ön İşleme:")
        self.preprocess_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        self.preprocess_var = tk.StringVar(value="basic")
        self.preprocess_combo = ttk.Combobox(self.control_frame, textvariable=self.preprocess_var)
        self.preprocess_combo['values'] = (
            "basic", "gray", "threshold", "adaptive_threshold", "otsu", 
            "gaussian_blur", "bilateral_filter", "dilation", "erosion", "opening", "closing"
        )
        self.preprocess_combo.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.preprocess_combo.bind("<<ComboboxSelected>>", self.update_preview)
        
        # Dil seçenekleri
        self.lang_label = ttk.Label(self.control_frame, text="Dil:")
        self.lang_label.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
        self.lang_var = tk.StringVar(value="eng")
        self.lang_combo = ttk.Combobox(self.control_frame, textvariable=self.lang_var, width=5)
        self.lang_combo['values'] = ("eng", "tur", "eng+tur")
        self.lang_combo.grid(row=0, column=4, padx=5, pady=5, sticky="w")
        
        # OCR butonu
        self.ocr_button = ttk.Button(self.control_frame, text="OCR Uygula", command=self.perform_ocr)
        self.ocr_button.grid(row=0, column=5, padx=5, pady=5, sticky="w")
        self.ocr_button.configure(state="disabled")
        
        # İlerleme çubuğu
        self.progress = ttk.Progressbar(self.control_frame, orient=tk.HORIZONTAL, length=100, mode='indeterminate')
        self.progress.grid(row=0, column=6, padx=5, pady=5, sticky="w")
        
        # Ön işleme parametreleri çerçevesi
        self.params_frame = ttk.LabelFrame(self.left_panel, text="Ön İşleme Parametreleri")
        self.params_frame.pack(fill=tk.X)
        
        # Eşik değeri
        self.threshold_label = ttk.Label(self.params_frame, text="Eşik Değeri:")
        self.threshold_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.threshold_var = tk.IntVar(value=127)
        self.threshold_scale = ttk.Scale(self.params_frame, from_=0, to=255, variable=self.threshold_var, 
                                         orient=tk.HORIZONTAL, length=200, command=self.update_preview)
        self.threshold_scale.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        self.threshold_value_label = ttk.Label(self.params_frame, text="127")
        self.threshold_value_label.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.threshold_var.trace_add("write", self.update_threshold_label)
        
        # Bulanıklaştırma boyutu
        self.blur_label = ttk.Label(self.params_frame, text="Bulanıklaştırma:")
        self.blur_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
        
        self.blur_var = tk.IntVar(value=5)
        self.blur_scale = ttk.Scale(self.params_frame, from_=1, to=21, variable=self.blur_var, 
                                   orient=tk.HORIZONTAL, length=200, command=self.update_preview)
        self.blur_scale.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        self.blur_value_label = ttk.Label(self.params_frame, text="5")
        self.blur_value_label.grid(row=1, column=2, padx=5, pady=5, sticky="w")
        self.blur_var.trace_add("write", self.update_blur_label)
        
        # Morfolojik işlem boyutu
        self.morph_label = ttk.Label(self.params_frame, text="Morfolojik Boyut:")
        self.morph_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        
        self.morph_var = tk.IntVar(value=3)
        self.morph_scale = ttk.Scale(self.params_frame, from_=1, to=21, variable=self.morph_var, 
                                    orient=tk.HORIZONTAL, length=200, command=self.update_preview)
        self.morph_scale.grid(row=2, column=1, padx=5, pady=5, sticky="w")
        
        self.morph_value_label = ttk.Label(self.params_frame, text="3")
        self.morph_value_label.grid(row=2, column=2, padx=5, pady=5, sticky="w")
        self.morph_var.trace_add("write", self.update_morph_label)
        
        # Sonuç çerçevesi
        self.result_frame = ttk.LabelFrame(self.right_panel, text="İşlenmiş Görüntü")
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.result_label = ttk.Label(self.result_frame)
        self.result_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # OCR sonuç çerçevesi
        self.ocr_result_frame = ttk.LabelFrame(self.right_panel, text="OCR Sonucu")
        self.ocr_result_frame.pack(fill=tk.BOTH, expand=True)
        
        self.ocr_text = ScrolledText(self.ocr_result_frame, wrap=tk.WORD, font=("Arial", 24))
        self.ocr_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Durum çubuğu
        self.status_var = tk.StringVar(value="Hazır")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Değişkenler
        self.original_image = None
        self.processed_image = None
        self.tk_image = None
        self.tk_processed = None
        
        # Örnek görüntü yükle
        self.load_sample_image()
    
    def load_sample_image(self):
        """Örnek bir metin görüntüsü oluştur"""
        # Örnek metin görüntüsü oluştur
        sample_dir = "../images"
        os.makedirs(sample_dir, exist_ok=True)
        
        sample_path = os.path.join(sample_dir, "sample_text.png")
        
        # Eğer örnek görüntü yoksa oluştur
        if not os.path.exists(sample_path):
            # Boş bir görüntü oluştur
            img = np.ones((400, 800, 3), dtype=np.uint8) * 255
            
            # Metin ekle
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, "OpenCV ve Tesseract ile OCR Ornegi", (50, 70), font, 1.2, (0, 0, 0), 2)
            cv2.putText(img, "Bu metin OCR ile tanınacak.", (50, 120), font, 1, (0, 0, 0), 2)
            cv2.putText(img, "Turkce karakterler: çğıöşü", (50, 170), font, 1, (0, 0, 0), 2)
            cv2.putText(img, "Rakamlar: 0123456789", (50, 220), font, 1, (0, 0, 0), 2)
            cv2.putText(img, "Ozel karakterler: !@#$%^&*()", (50, 270), font, 1, (0, 0, 0), 2)
            cv2.putText(img, "Farkli boyut: Kucuk ve BUYUK", (50, 320), font, 1, (0, 0, 0), 2)
            
            # Görüntüyü kaydet
            cv2.imwrite(sample_path, img)
            print(f"Örnek metin görüntüsü oluşturuldu: {sample_path}")
        
        # Örnek görüntüyü yükle
        self.load_image_from_path(sample_path)
    
    def load_image(self):
        """Dosya diyaloğu ile görüntü yükle"""
        file_path = filedialog.askopenfilename(
            title="Görüntü Seç",
            filetypes=[
                ("Görüntü Dosyaları", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("Tüm Dosyalar", "*.*")
            ]
        )
        
        if file_path:
            self.load_image_from_path(file_path)
    
    def load_image_from_path(self, file_path):
        """Belirtilen yoldan görüntü yükle"""
        try:
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                raise ValueError("Görüntü okunamadı")
            
            self.status_var.set(f"Görüntü yüklendi: {os.path.basename(file_path)}")
            self.ocr_button.configure(state="normal")
            
            # Görüntüyü göster
            self.display_image(self.original_image, self.image_label)
            
            # Ön işleme uygula
            self.update_preview()
            
        except Exception as e:
            messagebox.showerror("Hata", f"Görüntü yüklenirken hata oluştu: {str(e)}")
            self.status_var.set("Hata: Görüntü yüklenemedi")
    
    def display_image(self, cv_image, label, max_width=1000, max_height=800):
        """OpenCV görüntüsünü Tkinter etiketinde göster"""
        if cv_image is None:
            return
        
        # Görüntüyü yeniden boyutlandır
        h, w = cv_image.shape[:2]
        
        # En-boy oranını koru
        if w > max_width or h > max_height:
            scale = min(max_width / w, max_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            cv_image = cv2.resize(cv_image, (new_w, new_h))
        
        # BGR'den RGB'ye dönüştür
        if len(cv_image.shape) == 3:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # PIL görüntüsüne dönüştür
        pil_image = Image.fromarray(cv_image)
        
        # Tkinter görüntüsüne dönüştür
        tk_image = ImageTk.PhotoImage(pil_image)
        
        # Etiketi güncelle
        label.configure(image=tk_image)
        label.image = tk_image  # Referansı koru
    
    def update_threshold_label(self, *args):
        """Eşik değeri etiketini güncelle"""
        self.threshold_value_label.configure(text=str(self.threshold_var.get()))
    
    def update_blur_label(self, *args):
        """Bulanıklaştırma değeri etiketini güncelle"""
        # Bulanıklaştırma değerinin tek sayı olmasını sağla
        value = self.blur_var.get()
        if value % 2 == 0:
            value += 1
            self.blur_var.set(value)
        self.blur_value_label.configure(text=str(value))
    
    def update_morph_label(self, *args):
        """Morfolojik işlem değeri etiketini güncelle"""
        # Morfolojik işlem değerinin tek sayı olmasını sağla
        value = self.morph_var.get()
        if value % 2 == 0:
            value += 1
            self.morph_var.set(value)
        self.morph_value_label.configure(text=str(value))
    
    def update_preview(self, *args):
        """Ön işleme önizlemesini güncelle"""
        if self.original_image is None:
            return
        
        # Ön işleme uygula
        self.processed_image = self.preprocess_image(self.original_image)
        
        # İşlenmiş görüntüyü göster
        self.display_image(self.processed_image, self.result_label)
    
    def preprocess_image(self, image):
        """Seçilen ön işleme yöntemini uygula"""
        method = self.preprocess_var.get()
        threshold_value = self.threshold_var.get()
        blur_size = self.blur_var.get()
        if blur_size % 2 == 0:
            blur_size += 1  # Bulanıklaştırma boyutu tek sayı olmalı
        
        morph_size = self.morph_var.get()
        if morph_size % 2 == 0:
            morph_size += 1  # Morfolojik işlem boyutu tek sayı olmalı
        
        # Görüntünün bir kopyasını al
        result = image.copy()
        
        # Gri tonlamaya dönüştür (çoğu işlem için gerekli)
        if len(result.shape) == 3:
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        else:
            gray = result.copy()
        
        if method == "basic":
            # Temel işleme: Gri tonlama
            return gray
        
        elif method == "gray":
            # Sadece gri tonlama
            return gray
        
        elif method == "threshold":
            # Basit eşikleme
            _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
            return thresh
        
        elif method == "adaptive_threshold":
            # Adaptif eşikleme
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, blur_size, 11
            )
        
        elif method == "otsu":
            # Otsu eşikleme
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresh
        
        elif method == "gaussian_blur":
            # Gaussian bulanıklaştırma
            return cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        
        elif method == "bilateral_filter":
            # İki taraflı filtreleme
            return cv2.bilateralFilter(gray, blur_size, 75, 75)
        
        elif method == "dilation":
            # Genişletme (dilation)
            kernel = np.ones((morph_size, morph_size), np.uint8)
            return cv2.dilate(gray, kernel, iterations=1)
        
        elif method == "erosion":
            # Aşındırma (erosion)
            kernel = np.ones((morph_size, morph_size), np.uint8)
            return cv2.erode(gray, kernel, iterations=1)
        
        elif method == "opening":
            # Açma (opening) - Aşındırma sonrası genişletme
            kernel = np.ones((morph_size, morph_size), np.uint8)
            return cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        elif method == "closing":
            # Kapama (closing) - Genişletme sonrası aşındırma
            kernel = np.ones((morph_size, morph_size), np.uint8)
            return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Varsayılan olarak orijinal görüntüyü döndür
        return result
    
    def perform_ocr(self):
        """OCR işlemini gerçekleştir"""
        if self.processed_image is None:
            messagebox.showwarning("Uyarı", "İşlenecek görüntü bulunamadı!")
            return
        
        # OCR işlemini ayrı bir thread'de çalıştır
        threading.Thread(target=self._ocr_thread).start()
    
    def _ocr_thread(self):
        """OCR işlemini arka planda çalıştır"""
        try:
            # İlerleme çubuğunu başlat
            self.progress.start()
            self.ocr_button.configure(state="disabled")
            self.status_var.set("OCR işlemi çalışıyor...")
            
            # Dil seçeneğini al
            lang = self.lang_var.get()
            
            # OCR işlemini gerçekleştir
            start_time = time.time()
            
            # Görüntüyü PIL formatına dönüştür
            if len(self.processed_image.shape) == 3:
                pil_img = Image.fromarray(cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.fromarray(self.processed_image)
            
            # Tesseract OCR uygula
            text = pytesseract.image_to_string(pil_img, lang=lang)
            
            # Süreyi hesapla
            elapsed_time = time.time() - start_time
            
            # Sonuçları göster
            self.root.after(0, lambda: self._update_results(text, elapsed_time))
            
        except Exception as e:
            self.root.after(0, lambda: self._show_error(str(e)))
        finally:
            # İlerleme çubuğunu durdur
            self.root.after(0, self._reset_progress)
    
    def _update_results(self, text, elapsed_time):
        """OCR sonuçlarını güncelle"""
        # Metin alanını temizle
        self.ocr_text.delete(1.0, tk.END)
        
        # Sonuçları ekle
        self.ocr_text.insert(tk.END, text)
        
        # Durum çubuğunu güncelle
        self.status_var.set(f"OCR tamamlandı ({elapsed_time:.2f} saniye)")
    
    def _show_error(self, error_msg):
        """Hata mesajını göster"""
        messagebox.showerror("OCR Hatası", f"OCR işlemi sırasında hata oluştu: {error_msg}")
        self.status_var.set("Hata: OCR işlemi başarısız")
    
    def _reset_progress(self):
        """İlerleme çubuğunu sıfırla"""
        self.progress.stop()
        self.ocr_button.configure(state="normal")

def main():
    print("OpenCV OCR Uygulaması")
    print("-" * 25)
    
    # Tesseract'ın kurulu olup olmadığını kontrol et
    try:
        pytesseract.get_tesseract_version()
        print(f"Tesseract sürümü: {pytesseract.get_tesseract_version()}")
    except Exception as e:
        print(f"Hata: Tesseract bulunamadı veya çalıştırılamadı. Hata: {str(e)}")
        print("Lütfen Tesseract OCR'ı yükleyin ve yolunu doğru şekilde ayarlayın.")
        print("Windows için: https://github.com/UB-Mannheim/tesseract/wiki")
        print("Linux için: sudo apt-get install tesseract-ocr")
        print("macOS için: brew install tesseract")
        return
    
    # Tkinter uygulamasını başlat
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 