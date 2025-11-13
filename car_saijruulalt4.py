import cv2
import numpy as np
import pytesseract
from datetime import datetime
import os
from collections import deque
import tkinter as tk
from tkinter import filedialog

# Tesseract зам
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class VideoPlateDetector:
    def __init__(self):
        # Хавтас үүсгэх
        self.save_folder = "detected_plates"
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        
        # Дугаар илрүүлэх classifier
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
        )
        
        # Хүснэгт мэдээлэл
        self.detected_plates = []  # [{time, plate, confidence, image}]
        
        # Давхцах үед алгасах
        self.recent_plates = deque(maxlen=5)
        
        print("✅ Систем бэлэн!")
    
    def select_video(self):
        """Видео файл сонгох"""
        root = tk.Tk()
        root.withdraw()
        video_path = filedialog.askopenfilename(
            title="Видео сонгох",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        root.destroy()
        return video_path
    
    def is_duplicate(self, text):
        """Давхцсан эсэхийг шалгах"""
        if not text or len(text) < 3:
            return True
        return text in self.recent_plates
    
    def detect_plates(self, frame):
        """Дугаар илрүүлэх - сайжруулсан"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Contrast сайжруулах
        gray = cv2.equalizeHist(gray)
        
        # Олон хэмжээсээр хайх
        plates = self.cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,  # Илүү нарийвчлалтай
            minNeighbors=3,     # Илүү олон дугаар олох
            minSize=(60, 20),   # Бага дугаар ч таних
            maxSize=(400, 150)
        )
        return plates
    
    def preprocess_plate(self, plate_img):
        """Зураг МАШ сайн боловсруулах"""
        # RGB -> Gray
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()
        
        # 1. Том болгох (МААШГҮЙ ТОМ)
        h, w = gray.shape
        scale = 200 / h  # 200 pixel өндөр болгох
        new_w = int(w * scale)
        gray = cv2.resize(gray, (new_w, 200), interpolation=cv2.INTER_CUBIC)
        
        # 2. Denoise - хүчтэй
        gray = cv2.fastNlMeansDenoising(gray, h=10)
        
        # 3. Contrast өсгөх
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # 4. Sharpening
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        gray = cv2.filter2D(gray, -1, kernel)
        
        # 5. Morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # 6. Adaptive threshold (OTSU-аас дээр)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 7. Дахин denoise
        binary = cv2.medianBlur(binary, 3)
        
        return binary
    
    def recognize_text(self, plate_img):
        """OCR - өндөр нарийвчлалтай"""
        try:
            # Боловсруулах
            processed = self.preprocess_plate(plate_img)
            
            # Олон config туршина
            configs = [
                '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--oem 3 --psm 11 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            ]
            
            results = []
            for config in configs:
                try:
                    # Эх зураг
                    text1 = pytesseract.image_to_string(processed, config=config)
                    cleaned1 = self.clean_text(text1)
                    if cleaned1 and len(cleaned1) >= 4:
                        results.append(cleaned1)
                    
                    # Урвуу зураг (цагаан дээр хар)
                    inverted = cv2.bitwise_not(processed)
                    text2 = pytesseract.image_to_string(inverted, config=config)
                    cleaned2 = self.clean_text(text2)
                    if cleaned2 and len(cleaned2) >= 4:
                        results.append(cleaned2)
                except:
                    pass
            
            # Хамгийн их давтагдсан
            if results:
                from collections import Counter
                counter = Counter(results)
                best_text, count = counter.most_common(1)[0]
                
                # Итгэлтэй эсэхийг тооцох
                confidence = (count / len(results)) * 100 if results else 0
                
                return best_text, confidence
        except Exception as e:
            print(f"   ⚠️  OCR алдаа: {e}")
        
        return None, 0
    
    def clean_text(self, text):
        """Текст цэвэрлэх"""
        # Зөвхөн үсэг тоо
        cleaned = ''.join(c for c in text if c.isalnum())
        cleaned = cleaned.upper().strip()
        
        # Хэт богино эсвэл урт бол алгасах
        if len(cleaned) < 4 or len(cleaned) > 12:
            return None
        
        return cleaned
    
    def draw_table(self, frame):
        """Хүснэгт зурах - баруун талд"""
        h, w = frame.shape[:2]
        
        # Хүснэгтийн өргөн
        table_width = 350
        table_x = w - table_width
        
        # Хар дэвсгэр
        overlay = frame.copy()
        cv2.rectangle(overlay, (table_x, 0), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Гарчиг
        cv2.rectangle(frame, (table_x, 0), (w, 50), (0, 100, 200), -1)
        cv2.putText(frame, "TANISAN DUGAARUD", 
                   (table_x + 10, 32), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        
        # Хүснэгтийн толгой
        y = 60
        cv2.line(frame, (table_x, y), (w, y), (100, 100, 100), 2)
        cv2.putText(frame, "Tsag", (table_x + 10, y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(frame, "Dugaar", (table_x + 100, y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(frame, "%", (table_x + 280, y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        y += 35
        cv2.line(frame, (table_x, y), (w, y), (100, 100, 100), 1)
        
        # Мөрүүд харуулах (сүүлийн 12)
        start_idx = max(0, len(self.detected_plates) - 12)
        for i, detection in enumerate(self.detected_plates[start_idx:]):
            y += 40
            if y > h - 20:
                break
            
            time_str = detection['time'].strftime("%H:%M:%S")
            plate = detection['plate']
            conf = detection['confidence']
            
            # Цаг
            cv2.putText(frame, time_str, (table_x + 10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Дугаар (өнгөтэй)
            color = (0, 255, 0) if conf > 70 else (0, 200, 255) if conf > 40 else (0, 100, 255)
            cv2.putText(frame, plate, (table_x + 100, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            
            # Confidence
            cv2.putText(frame, f"{conf:.0f}", (table_x + 280, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Статистик доод хэсэгт
        cv2.rectangle(frame, (table_x, h - 60), (w, h), (40, 40, 40), -1)
        cv2.putText(frame, f"Niit: {len(self.detected_plates)}", 
                   (table_x + 10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 1)
        
        high_conf = sum(1 for d in self.detected_plates if d['confidence'] > 70)
        cv2.putText(frame, f"Ondor: {high_conf}", 
                   (table_x + 10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 1)
        
        return frame
    
    def draw_detection(self, frame, x, y, w, h, text, confidence):
        """Дугаар дээр хүрээ"""
        # Өнгө confidence-аас хамаарна
        if confidence > 70:
            color = (0, 255, 0)  # Ногоон
        elif confidence > 40:
            color = (0, 200, 255)  # Шар
        else:
            color = (0, 100, 255)  # Улаан
        
        # Хүрээ
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Дэвсгэр + текст
        if text:
            label = f"{text} ({confidence:.0f}%)"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            cv2.rectangle(frame, (x, y-30), (x + text_size[0] + 10, y), (0, 0, 0), -1)
            cv2.rectangle(frame, (x, y-30), (x + text_size[0] + 10, y), color, 2)
            
            cv2.putText(frame, label, (x + 5, y - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame
    
    def save_to_file(self):
        """Excel файл үүсгэх"""
        if not self.detected_plates:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_file = os.path.join(self.save_folder, f"report_{timestamp}.txt")
        
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("МАШИНЫ ДУГААР ТАНИХ - ТАЙЛАН\n")
            f.write("="*60 + "\n\n")
            f.write(f"Огноо: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Нийт танигдсан: {len(self.detected_plates)}\n\n")
            f.write("-"*60 + "\n")
            f.write(f"{'№':<5} {'Цаг':<12} {'Дугаар':<15} {'Итгэлт':<10}\n")
            f.write("-"*60 + "\n")
            
            for i, det in enumerate(self.detected_plates, 1):
                time_str = det['time'].strftime("%H:%M:%S")
                f.write(f"{i:<5} {time_str:<12} {det['plate']:<15} {det['confidence']:.1f}%\n")
        
        print(f"\n💾 Тайлан хадгалагдлаа: {txt_file}")

def main():
    print("\n" + "="*70)
    print(" "*15 + "🚗 ВИДЕО ДУГААР ТАНИХ (ХҮСНЭГТТЭЙ) 🚗")
    print("="*70)
    print("\n📌 Онцлог:")
    print("  ✓ Видео файлаас дугаар таних")
    print("  ✓ Баруун талд хүснэгт харуулах")
    print("  ✓ Өндөр нарийвчлалтай OCR")
    print("  ✓ Confidence хувь харуулах")
    print("  ✓ Тайлан үүсгэх")
    print("\n" + "-"*70 + "\n")
    
    detector = VideoPlateDetector()
    
    # Видео сонгох
    print("📁 Видео файл сонгох цонх нээгдэнэ...")
    video_path = detector.select_video()
    
    if not video_path:
        print("❌ Видео сонгогдсонгүй!")
        return
    
    print(f"✅ Видео: {os.path.basename(video_path)}\n")
    
    # Видео нээх
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Видео нээгдсэнгүй!")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"🎬 FPS: {fps} | Нийт frame: {total_frames}")
    print(f"⏱️  Үргэлжлэх хугацаа: {total_frames/fps:.1f} секунд\n")
    print("🚀 Таних эхэллээ...\n")
    print("   SPACE - Түр зогсоох/Үргэлжлүүлэх")
    print("   Q - Дуусгах\n")
    print("-"*70 + "\n")
    
    frame_count = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n✅ Видео дууслаа!")
                break
            
            frame_count += 1
            
            # Том resolution бол багасгах
            h, w = frame.shape[:2]
            if w > 1280:
                scale = 1280 / w
                frame = cv2.resize(frame, (1280, int(h * scale)))
            
            # 5 frame тутамд таних (хурд нэмэх)
            if frame_count % 5 == 0:
                plates = detector.detect_plates(frame)
                
                for (x, y, w_p, h_p) in plates:
                    plate_img = frame[y:y+h_p, x:x+w_p]
                    
                    # OCR
                    text, confidence = detector.recognize_text(plate_img)
                    
                    if text and not detector.is_duplicate(text):
                        # Хадгалах
                        detector.detected_plates.append({
                            'time': datetime.now(),
                            'plate': text,
                            'confidence': confidence,
                            'image': plate_img.copy()
                        })
                        detector.recent_plates.append(text)
                        
                        print(f"✅ Шинэ: {text} ({confidence:.0f}%) | Нийт: {len(detector.detected_plates)}")
                    
                    # Зурах
                    if text:
                        frame = detector.draw_detection(frame, x, y, w_p, h_p, text, confidence)
            
            # Хүснэгт зурах
            frame = detector.draw_table(frame)
            
            # Progress
            progress = (frame_count / total_frames) * 100
            cv2.putText(frame, f"Progress: {progress:.1f}%", 
                       (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 100), 2)
        
        cv2.imshow('Video Dugaar Tanikh', frame)
        
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord(' '):
            paused = not paused
            print("⏸️  Зогсоосон" if paused else "▶️  Үргэлжилж байна")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Тайлан хадгалах
    if detector.detected_plates:
        detector.save_to_file()
    
    # Дүгнэлт
    print("\n" + "="*70)
    print(" "*25 + "📊 ДҮГНЭЛТ")
    print("="*70)
    print(f"Нийт танигдсан: {len(detector.detected_plates)}")
    
    high_conf = sum(1 for d in detector.detected_plates if d['confidence'] > 70)
    med_conf = sum(1 for d in detector.detected_plates if 40 <= d['confidence'] <= 70)
    low_conf = sum(1 for d in detector.detected_plates if d['confidence'] < 40)
    
    print(f"Өндөр итгэлттэй (>70%): {high_conf}")
    print(f"Дунд итгэлттэй (40-70%): {med_conf}")
    print(f"Бага итгэлттэй (<40%): {low_conf}")
    
    print(f"\n💾 Файлууд: {detector.save_folder}/")
    print("\n👋 Баяртай!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()