import cv2
import numpy as np
import pytesseract
from datetime import datetime
import os
from collections import deque
import tkinter as tk
from tkinter import filedialog
import threading
import queue

# Tesseract зам
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class OptimizedPlateDetector:
    def __init__(self):
        self.save_folder = "detected_plates"
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
        )
        
        self.detected_plates = []
        self.recent_plates = {}
        
        # Шалгуур - ИЛҮҮ зөөлөн
        self.MIN_CONFIDENCE = 60  # 60% (өмнө 75)
        self.MIN_PLATE_SIZE = 60   # 60px (өмнө 80)
        self.FRAME_SKIP = 15       # 15 frame (өмнө 30)
        
        # OCR queue (thread-д ажиллуулах)
        self.ocr_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        print("✅ Оновчтой систем бэлэн!")
    
    def select_video(self):
        root = tk.Tk()
        root.withdraw()
        video_path = filedialog.askopenfilename(
            title="Видео сонгох",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("All files", "*.*")
            ]
        )
        root.destroy()
        return video_path
    
    def is_valid_plate(self, text):
        """Дугаар эсэхийг шалгах - зөөлөн"""
        if not text or len(text) < 4:
            return False
        
        # Ядаж 1 тоо байх
        has_digit = any(c.isdigit() for c in text)
        if not has_digit:
            return False
        
        # Зөвхөн үсэг тоо
        if not text.isalnum():
            return False
        
        # Урт шалгах
        if len(text) > 12:
            return False
        
        return True
    
    def is_duplicate(self, text, frame_number):
        if text in self.recent_plates:
            if frame_number - self.recent_plates[text] < self.FRAME_SKIP:
                return True
        self.recent_plates[text] = frame_number
        return False
    
    def detect_plates(self, frame):
        """Дугаар илрүүлэх - хурдан"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Хурдан шалгах
        plates = self.cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(self.MIN_PLATE_SIZE, 25)
        )
        
        # Харьцаа шалгах
        valid = []
        for (x, y, w, h) in plates:
            ratio = w / h
            if 1.5 <= ratio <= 6.0:
                valid.append((x, y, w, h))
        
        return valid
    
    def enhance_plate(self, plate_img):
        """Зураг ХУРДАН сайжруулах"""
        # Саарал
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()
        
        h, w = gray.shape
        
        # Томруулах - хурдан
        target_h = 150  # Багасгасан (өмнө 300)
        scale = target_h / h
        new_w = int(w * scale)
        gray = cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # Denoise - хурдан
        gray = cv2.fastNlMeansDenoising(gray, h=7)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Threshold - 2 аrга
        _, binary1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
        
        return [binary1, binary2, gray]
    
    def ocr_fast(self, images):
        """ХУРДАН OCR - зөвхөн хамгийн сайн config"""
        all_results = []
        
        # 1 config л ашиглах - хурдан
        config = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        for img in images:
            try:
                # Шууд текст авах
                text = pytesseract.image_to_string(img, config=config)
                cleaned = self.clean_text(text)
                
                if cleaned and self.is_valid_plate(cleaned):
                    # Confidence тооцох (data ашиглахгүй - хурдан)
                    # Урт болон тэмдэгтийн төрлөөс estimate хийх
                    digit_ratio = sum(c.isdigit() for c in cleaned) / len(cleaned)
                    conf = 60 + (digit_ratio * 30)  # 60-90%
                    all_results.append((cleaned, conf))
            except:
                pass
        
        if all_results:
            # Хамгийн урт текст сонгох (ихэвчлэн зөв байдаг)
            all_results.sort(key=lambda x: (len(x[0]), x[1]), reverse=True)
            return all_results[0]
        
        return None, 0
    
    def clean_text(self, text):
        """Цэвэрлэх"""
        cleaned = ''.join(c for c in text if c.isalnum())
        cleaned = cleaned.upper().strip()
        
        # I -> 1, O -> 0 засварлах
        cleaned = cleaned.replace('I', '1').replace('O', '0')
        
        if len(cleaned) < 4 or len(cleaned) > 12:
            return None
        
        return cleaned
    
    def draw_simple_table(self, frame):
        """Энгийн хүснэгт - хурдан зурах"""
        h, w = frame.shape[:2]
        table_w = 320
        table_x = w - table_w
        
        # Хар дэвсгэр
        cv2.rectangle(frame, (table_x, 0), (w, h), (20, 20, 20), -1)
        
        # Гарчиг
        cv2.rectangle(frame, (table_x, 0), (w, 50), (0, 100, 0), -1)
        cv2.putText(frame, "TANISAN", (table_x + 100, 35), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
        
        # Мөрүүд
        y = 70
        for i, det in enumerate(self.detected_plates[-8:], 1):  # Сүүлийн 8
            time_str = det['time'].strftime("%H:%M:%S")
            plate = det['plate']
            conf = det['confidence']
            
            color = (0, 255, 0) if conf >= 75 else (0, 200, 255)
            
            cv2.putText(frame, f"{time_str}", (table_x + 10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            
            cv2.putText(frame, plate, (table_x + 100, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            
            y += 35
        
        # Статистик
        cv2.rectangle(frame, (table_x, h-50), (w, h), (30, 30, 30), -1)
        cv2.putText(frame, f"Niit: {len(self.detected_plates)}", 
                   (table_x + 20, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
        
        return frame
    
    def draw_detection(self, frame, x, y, w, h, text, conf):
        """Дугаар дээр хүрээ"""
        color = (0, 255, 0) if conf >= 75 else (0, 200, 255)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        if text:
            cv2.rectangle(frame, (x, y-25), (x+200, y), (0, 0, 0), -1)
            cv2.putText(frame, f"{text}", (x+5, y-8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def create_result_window(self, plate_img, text, conf):
        """Үр дүн цонх - энгийн"""
        h, w = plate_img.shape[:2]
        scale = 200 / h
        enlarged = cv2.resize(plate_img, (int(w*scale), 200))
        
        window_h = 300
        window_w = max(enlarged.shape[1], 400)
        window = np.zeros((window_h, window_w, 3), dtype=np.uint8)
        window[:] = (25, 25, 25)
        
        # Зураг
        x_off = (window_w - enlarged.shape[1]) // 2
        window[20:220, x_off:x_off+enlarged.shape[1]] = enlarged
        
        # Текст
        color = (0, 255, 0) if conf >= 75 else (0, 200, 255)
        cv2.putText(window, text, (20, 260), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 3)
        cv2.putText(window, f"{conf:.0f}%", (20, 290), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
        
        return window
    
    def save_result(self, plate_img, text, conf):
        """Хадгалах"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        img_file = os.path.join(self.save_folder, f"{text}_{timestamp}.jpg")
        cv2.imwrite(img_file, plate_img)

def main():
    print("\n" + "="*70)
    print(" "*15 + "🚗 ВИДЕО ДУГААР ТАНИХ (ОНОВЧТОЙ) 🚗")
    print("="*70)
    print("\n⚡ Онцлог:")
    print("  • ХУРДАН ажиллана")
    print("  • Зөөлөн шалгуур (60%+)")
    print("  • Видео smooth харагдана")
    print("  • Энгийн боловч үр дүнтэй")
    print("\n" + "-"*70 + "\n")
    
    detector = OptimizedPlateDetector()
    
    print("📁 Видео сонгох...")
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
    
    # Мэдээлэл авах
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 Видео мэдээлэл:")
    print(f"   FPS: {fps}")
    print(f"   Resolution: {width}x{height}")
    print(f"   Frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.1f}s\n")
    
    # Display size тохируулах
    display_width = 1280 if width > 1280 else width
    display_height = int(height * (display_width / width))
    
    print("🚀 Эхэлж байна...\n")
    print("   SPACE - Зогсоох/Үргэлжлүүлэх")
    print("   Q - Дуусгах")
    print("   S - Screenshot авах\n")
    print("-"*70 + "\n")
    
    frame_count = 0
    paused = False
    processing_count = 0
    
    # FPS харуулах
    prev_time = cv2.getTickCount()
    fps_display = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n✅ Видео дууслаа!")
                break
            
            frame_count += 1
            
            # Display size-д багасгах
            if frame.shape[1] != display_width:
                frame = cv2.resize(frame, (display_width, display_height))
            
            # FPS тооцох
            curr_time = cv2.getTickCount()
            time_diff = (curr_time - prev_time) / cv2.getTickFrequency()
            if time_diff > 0:
                fps_display = 1.0 / time_diff
            prev_time = curr_time
            
            # 5 frame тутамд таних (илүү хурдан)
            if frame_count % 5 == 0:
                plates = detector.detect_plates(frame)
                
                for (x, y, w, h) in plates:
                    processing_count += 1
                    plate_img = frame[y:y+h, x:x+w]
                    
                    # ХУРДАН OCR
                    enhanced = detector.enhance_plate(plate_img)
                    text, conf = detector.ocr_fast(enhanced)
                    
                    # Зөвхөн сайн үр дүн
                    if text and conf >= detector.MIN_CONFIDENCE:
                        if not detector.is_duplicate(text, frame_count):
                            # Хадгалах
                            detector.detected_plates.append({
                                'time': datetime.now(),
                                'plate': text,
                                'confidence': conf
                            })
                            
                            detector.save_result(plate_img, text, conf)
                            
                            print(f"✅ {len(detector.detected_plates)}. {text} ({conf:.0f}%)")
                            
                            # Үр дүн цонх
                            result_win = detector.create_result_window(plate_img, text, conf)
                            cv2.imshow(f"Result - {text}", result_win)
                    
                    # Зурах
                    if text:
                        print(f"\n🚗 Дугаар #{i+1}: {text}")
                    else:
                        print(f"\ns  Дугаар #{i+1}: Танигдсангүй")
                        text = "Unknown"
        except Exception as e:
    print(f"\n Алдаа гарлаа: {e}")
text = "Error"

if __name__ == "__main__":
    main()