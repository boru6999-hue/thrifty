import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import pytesseract
import numpy as np
from PIL import Image, ImageTk
import os
import re

# ⚠️ Windows: Tesseract-ийн замыг зааж өгнө
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class PlateRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Машины дугаар таних систем - Сайжруулсан хувилбар")
        self.root.geometry("1000x650")
        self.root.resizable(False, False)

        self.current_image = None
        self.plate_text = ""
        self.photo = None
        self.confidence_score = 0

        # Дээд мөр
        header = tk.Label(root, text="🚙 Машины дугаар таних систем (v2.0)", 
                          font=("Arial", 16, "bold"), bg="#2c3e50", fg="white", pady=12)
        header.pack(fill="x")

        # Үндсэн хаалт
        main_frame = tk.Frame(root, bg="#ecf0f1")
        main_frame.pack(pady=10, padx=20, fill="both", expand=True)

        # Зүүн тал
        left_frame = tk.Frame(main_frame, width=320, padx=10, bg="#ecf0f1")
        left_frame.pack(side="left", fill="y")

        btn_frame = tk.Frame(left_frame, bg="#ecf0f1")
        btn_frame.pack(pady=10)

        self.btn_load = tk.Button(btn_frame, text="📁 Зураг сонгох", command=self.load_image,
                                  bg="#27ae60", fg="white", font=("Arial", 11, "bold"), 
                                  width=20, height=2, cursor="hand2", relief="flat")
        self.btn_load.pack(pady=5)

        self.btn_recognize = tk.Button(btn_frame, text="🔍 Дугаар таних", command=self.recognize_plate,
                                       bg="#3498db", fg="white", font=("Arial", 11, "bold"), 
                                       width=20, height=2, state="disabled", cursor="hand2", relief="flat")
        self.btn_recognize.pack(pady=5)

        self.btn_save = tk.Button(btn_frame, text="💾 Үр дүн хадгалах", command=self.save_result,
                                  bg="#9b59b6", fg="white", font=("Arial", 11, "bold"), 
                                  width=20, height=2, state="disabled", cursor="hand2", relief="flat")
        self.btn_save.pack(pady=5)

        self.btn_reset = tk.Button(btn_frame, text="🔄 Шинэчлэх", command=self.reset,
                                   bg="#e67e22", fg="white", font=("Arial", 11, "bold"), 
                                   width=20, height=2, cursor="hand2", relief="flat")
        self.btn_reset.pack(pady=5)

        # Үр дүн харуулах хэсэг
        result_frame = tk.LabelFrame(left_frame, text="📋 Танилтын үр дүн", 
                                     font=("Arial", 11, "bold"), padx=10, pady=10, bg="#ecf0f1")
        result_frame.pack(pady=15, fill="x")

        tk.Label(result_frame, text="Дугаар:", font=("Arial", 10), bg="#ecf0f1").pack(anchor="w")
        
        self.result_entry = tk.Entry(result_frame, font=("Arial", 18, "bold"), width=14, 
                                     justify="center", bg="white", relief="solid", 
                                     bd=2, fg="#2c3e50")
        self.result_entry.pack(pady=5)

        self.confidence_label = tk.Label(result_frame, text="", font=("Arial", 9), bg="#ecf0f1")
        self.confidence_label.pack(pady=3)

        self.format_label = tk.Label(result_frame, text="", font=("Arial", 9), 
                                     bg="#ecf0f1", wraplength=280)
        self.format_label.pack(pady=3)

        # Тохиргоо
        settings_frame = tk.LabelFrame(left_frame, text="⚙️ Тохиргоо", 
                                      font=("Arial", 10, "bold"), padx=10, pady=5, bg="#ecf0f1")
        settings_frame.pack(pady=10, fill="x")

        self.enhance_var = tk.BooleanVar(value=True)
        tk.Checkbutton(settings_frame, text="Дэвшилтэт боловсруулалт", 
                      variable=self.enhance_var, bg="#ecf0f1").pack(anchor="w")

        self.multi_attempt_var = tk.BooleanVar(value=True)
        tk.Checkbutton(settings_frame, text="Олон удаагийн танилт", 
                      variable=self.multi_attempt_var, bg="#ecf0f1").pack(anchor="w")

        # Зааварчилгаа
        info_frame = tk.LabelFrame(left_frame, text="ℹ️ Зөвлөмж", 
                                  font=("Arial", 10, "bold"), padx=10, pady=5, bg="#ecf0f1")
        info_frame.pack(pady=10, fill="x")
        
        info_text = ("• Тод, ойрын зураг сонгоно\n"
                    "• Дугаар тодорхой харагдаж байх\n"
                    "• Сайн гэрэлтүүлэгтэй зураг\n"
                    "• JPG, PNG форматтай")
        tk.Label(info_frame, text=info_text, font=("Arial", 8), 
                justify="left", anchor="w", bg="#ecf0f1").pack(anchor="w")

        # Баруун тал: Зураг харуулах
        canvas_frame = tk.Frame(main_frame, bg="#ecf0f1")
        canvas_frame.pack(side="right", padx=10, fill="both", expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, width=600, height=500, bg="#ffffff", 
                               relief="solid", bd=1)
        self.canvas.pack()
        
        self.canvas.create_text(300, 250, text="Зураг сонгоно уу", 
                               font=("Arial", 14), fill="#95a5a6")

        # Статус бар
        self.status = tk.Label(root, text="✅ Бэлэн. Зураг сонгоно уу.", 
                               relief="sunken", anchor="w", font=("Arial", 9), bg="#34495e", fg="white")
        self.status.pack(side="bottom", fill="x")

    def reset(self):
        """Бүх зүйлийг анхны байдалд оруулах"""
        self.current_image = None
        self.plate_text = ""
        self.photo = None
        self.confidence_score = 0
        self.result_entry.delete(0, tk.END)
        self.confidence_label.config(text="")
        self.format_label.config(text="")
        self.btn_recognize.config(state="disabled")
        self.btn_save.config(state="disabled")
        self.canvas.delete("all")
        self.canvas.create_text(300, 250, text="Зураг сонгоно уу", 
                               font=("Arial", 14), fill="#95a5a6")
        self.status.config(text="✅ Бэлэн. Зураг сонгоно уу.")

    def load_image(self):
        """Зураг сонгох"""
        file_path = filedialog.askopenfilename(
            title="Машины зураг сонгох",
            filetypes=[("Зургийн файлууд", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not file_path:
            return

        if not os.path.exists(file_path):
            messagebox.showerror("Алдаа", "Файл олдсонгүй!")
            return

        try:
            img = cv2.imread(file_path)
            if img is None:
                messagebox.showerror("Алдаа", "Зургийг уншиж чадсангүй.")
                return

            # Зургийг багасгах (performance-ийн төлөө)
            max_dim = 1200
            h, w = img.shape[:2]
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            img_pil.thumbnail((600, 500), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(img_pil)

            self.canvas.delete("all")
            self.canvas.create_image(300, 250, anchor="center", image=self.photo)
            
            self.current_image = img.copy()
            self.btn_recognize.config(state="normal")
            self.result_entry.delete(0, tk.END)
            self.confidence_label.config(text="")
            self.format_label.config(text="")
            self.status.config(text=f"📂 {os.path.basename(file_path)} - {img.shape[1]}x{img.shape[0]}px")
            
        except Exception as e:
            messagebox.showerror("Алдаа", f"Зураг ачаалахад алдаа:\n{str(e)}")
            self.status.config(text="❌ Зураг ачаалахад алдаа")

    def preprocess_plate(self, plate_img, method=1):
        """Дугаарын хэсгийг боловсруулах - олон аргатай"""
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        if method == 1:
            # Арга 1: CLAHE + Adaptive Threshold
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            blurred = cv2.bilateralFilter(enhanced, 11, 17, 17)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 11, 2)
        elif method == 2:
            # Арга 2: Otsu + Morphology
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # Арга 3: Simple threshold
            blurred = cv2.medianBlur(gray, 5)
            _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        
        # Чимээ цэвэрлэх
        kernel = np.ones((1, 1), np.uint8)
        clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)
        
        return clean

    def find_plate_regions(self, img):
        """Дугаар байж болох бүх бүсийг олох"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Олон төрлийн edge detection
        edges1 = cv2.Canny(gray, 50, 150)
        edges2 = cv2.Canny(gray, 100, 200)
        edges = cv2.bitwise_or(edges1, edges2)
        
        # Морфологи
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        plate_candidates = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # Хэт жижиг бүс орхих
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Машины дугаарын харьцаа: 1.5 - 7.0
            # Монгол дугаар: ~3-5, Европ: ~4-5
            if 1.5 <= aspect_ratio <= 7.0 and area > 500:
                extent = area / (w * h) if (w * h) > 0 else 0
                if extent > 0.3:  # Бүс хангалттай дүүрэн эсэх
                    plate_candidates.append({
                        'contour': cnt,
                        'bbox': (x, y, w, h),
                        'area': area,
                        'ratio': aspect_ratio,
                        'score': area * extent  # Эрэмбэлэхэд ашиглах
                    })
        
        # Хамгийн сайн кандидатуудыг буцаах
        plate_candidates.sort(key=lambda x: x['score'], reverse=True)
        return plate_candidates[:5]  # Топ 5 кандидат

    def validate_plate_format(self, text):
        """Монгол/Олон улсын дугаарын формат шалгах"""
        if not text or len(text) < 4:
            return False, "Хэт богино", 0
        # Монгол дугаарын формат: УБ1234АА, 1234УБА, гм
        # Each tuple is (regex_pattern, description, score)
        patterns = [
            (r'^[А-ЯӨҮ]{2}\d{4}[А-ЯӨҮ]{2}$', "Монгол стандарт (УБ1234АА)", 95),
            (r'^\d{4}[А-ЯӨҮ]{3}$', "Монгол 2 (1234УБА)", 90),
            (r'^[A-Z]{2}\d{4}[A-Z]{2}$', "Олон улсын (AB1234CD)", 85),
            (r'^[A-Z]{1,3}\d{3,4}$', "Товч формат (ABC123)", 75),
            (r'^\d{4}[A-Z]{2,3}$', "Тоо + үсэг (1234AB)", 70),
        ]

        for pattern, desc, score in patterns:
            try:
                if re.match(pattern, text):
                    return True, desc, score
            except re.error:
                # In the unlikely event of a bad regex, skip it
                continue
        
        # Бусад тохиолдолд үсэг-тооны харьцааг шалгах
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        
        if letters >= 2 and digits >= 3:
            return True, "Таамаглал (үсэг + тоо)", 60
        
        return False, "Буруу формат", 40

    def clean_ocr_text(self, text):
        """OCR-ийн алдаа засах"""
        # Энгийн засвар: том үсэг рүү хөрвүүлэх, тусгай тэмдэг устгах
        cleaned = re.sub(r'[^A-ZА-ЯӨҮЁІ0-9]', '', text.upper())

        # Удаан зөрөхөөс зайлсхийх үүднээс нэг талын бодлого ашиглана
        # Ерөнхийдээ OCR нь зарим үсгүүдийг тоогоор алдаж уншина (O->0, I->1, S->5, B->8, G->6)
        # Гэхдээ зарим тэмдэг нь жинхэнэ үсэг байж болно. Энд бид зөвлөмжийн дагуу зөвлөлдсөн
        # солилцоог хийдэг: үсэгүүдийг тоонууд руу (ихэнхдээ дугаарын төвшинд ашиглагдах) хөрвүүлэх.
        corrections = {
            'O': '0',
            'I': '1',
            'S': '5',
            'B': '8',
            'G': '6'
        }

        result_chars = []
        for ch in cleaned:
            if ch in corrections:
                # Хэрэв тэмдэг нь зөвхөн цифр байх ёстой болов уу гэж таамаглах бол орлуулна.
                # Энгийн heuristic: хэрвээ мөрөнд аль хэдийн цифр байгаа бол үсгийг цифрт хувирга.
                if any(c.isdigit() for c in cleaned):
                    result_chars.append(corrections[ch])
                else:
                    result_chars.append(ch)
            else:
                result_chars.append(ch)

        return ''.join(result_chars)

    def ocr_with_multiple_configs(self, processed_img):
        """Олон төрлийн OCR тохиргоогоор танилт хийх"""
        results = []
        
        # PSM (Page Segmentation Mode) төрлүүд
        psm_modes = [
            7,   # Дан мөр (дугаарт тохиромжтой)
            8,   # Дан үг
            11,  # Эмх цэгцгүй текст
            13,  # Түүхий мөр
        ]
        
        configs = [
            '--oem 3 --psm {} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZАБВГДЕЁЖЗИЙКЛМНОӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯ0123456789',
            '--oem 3 --psm {} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            '--oem 1 --psm {}',
        ]
        
        # Allow both Latin and Cyrillic (where available) by requesting 'eng+rus' to Tesseract.
        # This is not guaranteed to be installed on the user's machine, but is a reasonable default
        # for plates that may contain Cyrillic letters.
        tess_lang = 'eng+rus'

        for psm in psm_modes:
            for config_template in configs:
                try:
                    config = config_template.format(psm)
                    text = pytesseract.image_to_string(processed_img, config=config, lang=tess_lang)
                    cleaned = self.clean_ocr_text(text)

                    if len(cleaned) >= 4:
                        valid, format_type, score = self.validate_plate_format(cleaned)
                        if valid or len(cleaned) <= 12:  # Хэт урт бол орхих
                            results.append({
                                'text': cleaned,
                                'score': score,
                                'format': format_type
                            })
                except Exception:
                    # Skip invalid configs or OCR failures silently
                    continue
        
        return results

    def recognize_plate(self):
        """Дугаар таних үндсэн функц - сайжруулсан"""
        if self.current_image is None:
            messagebox.showwarning("Анхааруулга", "Эхлээд зураг сонгоно уу.")
            return

        self.status.config(text="⏳ Танилт хийж байна... түр хүлээнэ үү.")
        self.root.update()

        try:
            img = self.current_image.copy()
            best_result = None
            best_score = 0
            best_bbox = None
            
            # Дугаарын бүс олох
            candidates = self.find_plate_regions(img)
            
            if not candidates and self.enhance_var.get():
                self.status.config(text="⏳ Дэвшилтэт хайлт хийж байна...")
                self.root.update()
                
                # Зургийг өөрчлөн дахин оролдох
                enhanced = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
                candidates = self.find_plate_regions(enhanced)
            
            # Бүх кандидат бүсээс OCR хийх
            all_results = []
            
            if candidates:
                for i, candidate in enumerate(candidates[:3]):  # Топ 3 кандидат
                    x, y, w, h = candidate['bbox']
                    
                    # Жижиг зай нэмэх
                    padding = 5
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(img.shape[1] - x, w + 2*padding)
                    h = min(img.shape[0] - y, h + 2*padding)
                    
                    roi = img[y:y+h, x:x+w]
                    
                    if self.multi_attempt_var.get():
                        # Олон аргаар боловсруулах
                        for method in [1, 2, 3]:
                            processed = self.preprocess_plate(roi, method)
                            ocr_results = self.ocr_with_multiple_configs(processed)
                            
                            for result in ocr_results:
                                result['bbox'] = (x, y, w, h)
                                result['method'] = method
                                all_results.append(result)
                    else:
                        processed = self.preprocess_plate(roi)
                        ocr_results = self.ocr_with_multiple_configs(processed)
                        for result in ocr_results:
                            result['bbox'] = (x, y, w, h)
                            all_results.append(result)
            
            # Бүх зургаас шууд OCR (fallback)
            if not all_results or len(all_results) < 3:
                processed_full = self.preprocess_plate(img)
                full_results = self.ocr_with_multiple_configs(processed_full)
                # Adjust each result's score and add once to all_results
                for result in full_results:
                    result['score'] = result.get('score', 0) - 10  # Бага оноо өгөх
                    all_results.append(result)
            
            # Хамгийн сайн үр дүн сонгох
            if all_results:
                all_results.sort(key=lambda x: x['score'], reverse=True)
                best_result = all_results[0]
                
                self.plate_text = best_result['text']
                self.confidence_score = best_result['score']
                
                self.result_entry.delete(0, tk.END)
                self.result_entry.insert(0, self.plate_text)
                self.result_entry.config(fg="#27ae60")
                
                self.confidence_label.config(
                    text=f"🎯 Итгэлцэл: {self.confidence_score}%",
                    fg="#27ae60" if self.confidence_score > 80 else "#f39c12"
                )
                
                self.format_label.config(
                    text=f"📋 Формат: {best_result['format']}",
                    fg="#2c3e50"
                )
                
                self.status.config(text=f"✅ Амжилттай! {len(all_results)} үр дүнгээс сонгосон")
                self.btn_save.config(state="normal")
                
                # Зураг дээр үр дүн харуулах
                if 'bbox' in best_result and best_result['bbox']:
                    x, y, w, h = best_result['bbox']
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(img, self.plate_text, (x, y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            else:
                self.result_entry.delete(0, tk.END)
                self.result_entry.insert(0, "Танигдаагүй")
                self.result_entry.config(fg="#e74c3c")
                self.confidence_label.config(text="❌ Дугаар олдсонгүй", fg="#e74c3c")
                self.format_label.config(text="💡 Илүү тод зураг оролдоно уу")
                self.status.config(text="⚠️ Дугаар танигдсангүй")
            
            # Үр дүнгийн зургийг харуулах
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_pil.thumbnail((600, 500), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(img_pil)
            
            self.canvas.delete("all")
            self.canvas.create_image(300, 250, anchor="center", image=self.photo)

        except Exception as e:
            messagebox.showerror("Алдаа", f"Танилтын алдаа:\n{str(e)}")
            self.status.config(text="❌ Алдаа гарлаа")
            print(f"DEBUG: {e}")
            import traceback
            traceback.print_exc()

    def save_result(self):
        """Үр дүнг текст файлд хадгалах"""
        if not self.plate_text:
            messagebox.showwarning("Анхааруулга", "Хадгалах үр дүн алга.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Текст файл", "*.txt"), ("Бүх файл", "*.*")],
            initialfile=f"plate_{self.plate_text}.txt"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"Машины дугаар: {self.plate_text}\n")
                    f.write(f"Итгэлцэл: {self.confidence_score}%\n")
                    from datetime import datetime
                    f.write(f"Огноо: {datetime.now().isoformat(sep=' ', timespec='seconds')}\n")
                messagebox.showinfo("Амжилттай", f"Файл хадгалагдлаа:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Алдаа", f"Хадгалах алдаа:\n{str(e)}")

# 🚀 Програм эхлүүлэх
if __name__ == "__main__":
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    if not os.path.exists(tesseract_path):
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Tesseract OCR олдсонгүй",
            "Tesseract OCR суугаагүй байна!\n\n"
            "Суулгах:\n"
            "1. https://github.com/UB-Mannheim/tesseract/wiki\n"
            "2. tesseract-ocr-w64-setup-*.exe татаж суулгана\n"
            "3. C:\\Program Files\\Tesseract-OCR\\ руу суулгана\n"
            "4. Програмыг дахин ажиллуулна"
        )
        root.destroy()
    else:
        root = tk.Tk()
        app = PlateRecognitionApp(root)
        root.mainloop()