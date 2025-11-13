import cv2
import numpy as np
import pytesseract
from datetime import datetime
import os
from collections import deque
import tkinter as tk
from tkinter import filedialog
import re

# Tesseract зам
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class StrictPlateDetector:
    def __init__(self):
        # Хавтас үүсгэх
        self.save_folder = "detected_plates"
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        
        # Classifier
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
        )
        
        # Хүснэгт мэдээлэл
        self.detected_plates = []
        
        # Сүүлийн дугаарууд (давхцах үед алгасах)
        self.recent_plates = {}  # {text: last_frame_number}
        
        # ХАТУУ ШАЛГУУР
        self.MIN_CONFIDENCE = 75  # Доод итгэл 75%
        self.MIN_PLATE_SIZE = 80   # Дугаарын доод хэмжээ (pixel)
        self.FRAME_SKIP = 30       # 30 frame алгасах (давхцах үед)
        
        print("✅ Хатуу шалгуурын систем бэлэн!")
    
    def select_video(self):
        """Видео сонгох"""
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
    
    def is_valid_plate_format(self, text):
        """Дугаарын формат зөв эсэхийг шалгах"""
        if not text or len(text) < 5 or len(text) > 10:
            return False
        
        # Хамгийн багадаа 2 тоо, 2 үсэг байх ёстой
        digit_count = sum(c.isdigit() for c in text)
        letter_count = sum(c.isalpha() for c in text)
        
        if digit_count < 2 or letter_count < 1:
            return False
        
        # Зөвхөн үсэг тоо байх
        if not text.isalnum():
            return False
        
        # Монгол дугаарын pattern (жишээ: УБ1234АА, 1234УБА гэх мэт)
        # Энэ хэсгийг өөрийн улсын дугаарын format-д тохируулж болно
        
        return True
    
    def is_duplicate(self, text, frame_number):
        """Давхцсан эсэхийг frame number-аар шалгах"""
        if text in self.recent_plates:
            last_frame = self.recent_plates[text]
            if frame_number - last_frame < self.FRAME_SKIP:
                return True
        
        self.recent_plates[text] = frame_number
        return False
    
    def detect_plates(self, frame):
        """Дугаар илрүүлэх - зөвхөн том, тодорхой дугаарыг"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Contrast сайжруулах
        gray = cv2.equalizeHist(gray)
        
        # Илрүүлэх - ХАТУУ параметр
        plates = self.cascade.detectMultiScale(
            gray, 
            scaleFactor=1.08,      # Бага scale = бага дугаар
            minNeighbors=5,        # Илүү итгэлтэй
            minSize=(self.MIN_PLATE_SIZE, 30),  # Доод хэмжээ
            maxSize=(500, 200)
        )
        
        # Дугаарын харьцааг шалгах (өргөн/өндөр = 2-5 орчим)
        valid_plates = []
        for (x, y, w, h) in plates:
            aspect_ratio = w / h
            if 2.0 <= aspect_ratio <= 5.5:  # Дугаарын хэвийн харьцаа
                valid_plates.append((x, y, w, h))
        
        return valid_plates
    
    def preprocess_plate(self, plate_img):
        """ХАМГИЙН сайн боловсруулалт"""
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()
        
        h, w = gray.shape
        
        # 1. АСАР ТОМ болгох (300px өндөр)
        scale = 300 / h
        new_w = int(w * scale)
        gray = cv2.resize(gray, (new_w, 300), interpolation=cv2.INTER_LANCZOS4)
        
        # 2. Bilateral filter (края хадгалж noise арилгах)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # 3. CLAHE - хүчтэй contrast
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # 4. Unsharp masking (тод болгох)
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        gray = cv2.addWeighted(gray, 2.0, gaussian, -1.0, 0)
        
        # 5. Морфологи
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # 6. Adaptive threshold
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 15, 2
        )
        
        # 7. Эцсийн denoise
        binary = cv2.medianBlur(binary, 3)
        
        return binary, gray
    
    def recognize_text_strict(self, plate_img):
        """МААШГҮЙ ХАТУУ OCR"""
        try:
            # Боловсруулах
            binary, gray = self.preprocess_plate(plate_img)
            
            # Tesseract config - зөвхөн үсэг тоо
            configs = [
                '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя',
                '--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя',
                '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя',
            ]
            
            all_results = []
            
            # Олон аргаар туршина
            test_images = [
                binary,
                cv2.bitwise_not(binary),  # Урвуу
                gray,  # Саарал
            ]
            
            for img in test_images:
                for config in configs:
                    try:
                        # OCR хийх
                        data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
                        
                        # Итгэлтэй текст авах
                        texts = []
                        confidences = []
                        
                        for i in range(len(data['text'])):
                            conf = int(data['conf'][i])
                            text = data['text'][i].strip()
                            
                            if conf > 60 and text:  # 60%-аас дээш итгэл
                                cleaned = self.clean_text(text)
                                if cleaned:
                                    texts.append(cleaned)
                                    confidences.append(conf)
                        
                        # Бүх текстийг нэгтгэх
                        if texts:
                            full_text = ''.join(texts)
                            avg_conf = sum(confidences) / len(confidences)
                            
                            if self.is_valid_plate_format(full_text):
                                all_results.append((full_text, avg_conf))
                    
                    except:
                        pass
            
            # Хамгийн сайн үр дүн сонгох
            if all_results:
                # Итгэл ба давтамжаар эрэмбэлэх
                from collections import Counter
                
                # Текстүүдийг тоолох
                text_counts = Counter([r[0] for r in all_results])
                
                # Хамгийн их давтагдсан, өндөр итгэлтэй текст
                best_candidates = []
                for text, count in text_counts.most_common(3):
                    # Энэ текстийн дундаж итгэл
                    confs = [r[1] for r in all_results if r[0] == text]
                    avg_conf = sum(confs) / len(confs)
                    best_candidates.append((text, avg_conf, count))
                
                # Эрэмбэлэх: давтамж * итгэл
                best_candidates.sort(key=lambda x: x[1] * x[2], reverse=True)
                
                if best_candidates:
                    best_text, best_conf, _ = best_candidates[0]
                    
                    # ХАТУУ шалгуур: 75%-аас дээш байх ёстой
                    if best_conf >= self.MIN_CONFIDENCE:
                        return best_text, best_conf, binary
                    
        except Exception as e:
            print(f"   ⚠️  OCR алдаа: {e}")
        
        return None, 0, None
    
    def clean_text(self, text):
        """Текст цэвэрлэх - хатуу"""
        # Зөвхөн үсэг тоо
        cleaned = ''.join(c for c in text if c.isalnum())
        cleaned = cleaned.upper().strip()
        
        # Богино эсвэл урт бол үгүй
        if len(cleaned) < 5 or len(cleaned) > 10:
            return None
        
        return cleaned
    
    def create_enlarged_view(self, plate_img, processed_img, text, confidence):
        """Том харуулах цонх - гоё"""
        # Зургуудыг томруулах
        h, w = plate_img.shape[:2]
        scale = 250 / h
        new_w = int(w * scale)
        
        enlarged_orig = cv2.resize(plate_img, (new_w, 250))
        
        if processed_img is not None:
            # Processed зургийг RGB болгох
            processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
            processed_rgb = cv2.resize(processed_rgb, (new_w, 250))
        else:
            processed_rgb = enlarged_orig.copy()
        
        # Цонхны хэмжээ
        window_h = 250 * 2 + 150  # 2 зураг + мэдээлэл
        window_w = max(new_w, 500)
        
        # Хар дэвсгэр
        window = np.zeros((window_h, window_w, 3), dtype=np.uint8)
        window[:] = (20, 20, 20)
        
        # Гарчиг
        cv2.rectangle(window, (0, 0), (window_w, 50), (0, 100, 200), -1)
        cv2.putText(window, "TANISAN DUGAAR", 
                   (window_w//2 - 120, 35), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        
        # Эх зураг
        y_pos = 60
        x_offset = (window_w - new_w) // 2
        window[y_pos:y_pos+250, x_offset:x_offset+new_w] = enlarged_orig
        cv2.putText(window, "Original", (x_offset + 10, y_pos + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
        
        # Боловсруулсан зураг
        y_pos += 260
        window[y_pos:y_pos+250, x_offset:x_offset+new_w] = processed_rgb
        cv2.putText(window, "Processed", (x_offset + 10, y_pos + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
        
        # Мэдээлэл
        y_pos += 270
        
        # Дугаар - ТОМ
        if text:
            color = (0, 255, 0) if confidence >= 85 else (0, 200, 255)
            cv2.putText(window, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_DUPLEX, 2, color, 3)
            
            # Итгэл
            conf_text = f"Confidence: {confidence:.1f}%"
            cv2.putText(window, conf_text, (20, y_pos + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Огноо
            time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(window, time_now, (20, y_pos + 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
        
        return window
    
    def draw_table(self, frame):
        """Хүснэгт зурах"""
        h, w = frame.shape[:2]
        table_width = 380
        table_x = w - table_width
        
        # Дэвсгэр
        overlay = frame.copy()
        cv2.rectangle(overlay, (table_x, 0), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        # Гарчиг
        cv2.rectangle(frame, (table_x, 0), (w, 60), (0, 120, 0), -1)
        cv2.putText(frame, "BURTGEL", 
                   (table_x + 120, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        
        # Толгой
        y = 75
        cv2.line(frame, (table_x + 10, y), (w - 10, y), (100, 100, 100), 2)
        y += 30
        
        cv2.putText(frame, "#", (table_x + 15, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "Tsag", (table_x + 50, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "Dugaar", (table_x + 140, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "Conf", (table_x + 290, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        y += 5
        cv2.line(frame, (table_x + 10, y), (w - 10, y), (100, 100, 100), 1)
        
        # Мөрүүд (сүүлийн 10)
        start_idx = max(0, len(self.detected_plates) - 10)
        for i, detection in enumerate(self.detected_plates[start_idx:], start=start_idx+1):
            y += 45
            if y > h - 100:
                break
            
            time_str = detection['time'].strftime("%H:%M:%S")
            plate = detection['plate']
            conf = detection['confidence']
            
            color = (0, 255, 0) if conf >= 85 else (0, 220, 220)
            
            # Дугаар
            cv2.putText(frame, f"{i}", (table_x + 15, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            # Цаг
            cv2.putText(frame, time_str, (table_x + 50, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            
            # Дугаар
            cv2.putText(frame, plate, (table_x + 140, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Confidence
            cv2.putText(frame, f"{conf:.0f}%", (table_x + 290, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # Статистик
        cv2.rectangle(frame, (table_x, h - 70), (w, h), (30, 30, 30), -1)
        cv2.putText(frame, f"Niit olson: {len(self.detected_plates)}", 
                   (table_x + 20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
        
        if self.detected_plates:
            avg_conf = sum(d['confidence'] for d in self.detected_plates) / len(self.detected_plates)
            cv2.putText(frame, f"Dundaj confidence: {avg_conf:.1f}%", 
                       (table_x + 20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
        
        return frame
    
    def save_detection(self, plate_img, processed_img, text, confidence):
        """Хадгалах"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Эх зураг
        img_file = os.path.join(self.save_folder, f"plate_{timestamp}.jpg")
        cv2.imwrite(img_file, plate_img)
        
        # Боловсруулсан зураг
        if processed_img is not None:
            proc_file = os.path.join(self.save_folder, f"processed_{timestamp}.jpg")
            cv2.imwrite(proc_file, processed_img)
        
        # Текст
        txt_file = os.path.join(self.save_folder, f"plate_{timestamp}.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"Огноо: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Дугаар: {text}\n")
            f.write(f"Итгэл: {confidence:.1f}%\n")

def main():
    print("\n" + "="*70)
    print(" "*12 + "🚗 ВИДЕО ДУГААР ТАНИХ (ӨНДӨР НАРИЙВЧЛАЛ) 🚗")
    print("="*70)
    print("\n⚠️  ХАТУУ ШАЛГУУР:")
    print(f"  • Доод итгэл: 75%")
    print(f"  • Доод хэмжээ: 80px")
    print(f"  • Зөвхөн зөв формат")
    print(f"  • Давхцал: 30 frame")
    print("\n" + "-"*70 + "\n")
    
    detector = StrictPlateDetector()
    
    # Видео сонгох
    print("📁 Видео файл сонгоно уу...")
    video_path = detector.select_video()
    
    if not video_path:
        print("❌ Видео сонгогдсонгүй!")
        return
    
    print(f"✅ Видео: {os.path.basename(video_path)}\n")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Видео нээгдсэнгүй!")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"🎬 FPS: {fps} | Frame: {total_frames} | Үргэлжлэх: {total_frames/fps:.1f}s\n")
    print("🚀 Эхлүүлж байна...\n")
    print("   SPACE - Зогсоох/Үргэлжлүүлэх")
    print("   Q - Дуусгах\n")
    print("-"*70 + "\n")
    
    frame_count = 0
    paused = False
    detection_windows = {}
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n✅ Видео дууслаа!")
                break
            
            frame_count += 1
            
            # Resolution
            h, w = frame.shape[:2]
            if w > 1280:
                scale = 1280 / w
                frame = cv2.resize(frame, (1280, int(h * scale)))
            
            # 2 frame тутамд таних (илүү нарийвчлалтай)
            if frame_count % 2 == 0:
                plates = detector.detect_plates(frame)
                
                for (x, y, w_p, h_p) in plates:
                    plate_img = frame[y:y+h_p, x:x+w_p]
                    
                    # ХАТУУ OCR
                    text, confidence, processed = detector.recognize_text_strict(plate_img)
                    
                    # ЗӨВХӨН өндөр итгэлтэй, давхцаагүй
                    if text and confidence >= detector.MIN_CONFIDENCE:
                        if not detector.is_duplicate(text, frame_count):
                            # Хадгалах
                            detector.detected_plates.append({
                                'time': datetime.now(),
                                'plate': text,
                                'confidence': confidence
                            })
                            
                            detector.save_detection(plate_img, processed, text, confidence)
                            
                            print(f"✅ #{len(detector.detected_plates)}: {text} ({confidence:.1f}%)")
                            
                            # Том харуулах цонх
                            enlarged = detector.create_enlarged_view(
                                plate_img, processed, text, confidence
                            )
                            window_name = f"Dugaar #{len(detector.detected_plates)} - {text}"
                            cv2.imshow(window_name, enlarged)
                            detection_windows[window_name] = True
                    
                    # Зурах (зөвхөн бодит дугаарыг)
                    if text and confidence >= detector.MIN_CONFIDENCE:
                        color = (0, 255, 0) if confidence >= 85 else (0, 200, 255)
                        cv2.rectangle(frame, (x, y), (x+w_p, y+h_p), color, 3)
                        
                        label = f"{text} ({confidence:.0f}%)"
                        cv2.rectangle(frame, (x, y-35), (x + 250, y), (0, 0, 0), -1)
                        cv2.rectangle(frame, (x, y-35), (x + 250, y), color, 2)
                        cv2.putText(frame, label, (x + 5, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Хүснэгт
            frame = detector.draw_table(frame)
            
            # Progress
            progress = (frame_count / total_frames) * 100
            cv2.rectangle(frame, (5, frame.shape[0] - 35), (300, frame.shape[0] - 5), (40, 40, 40), -1)
            cv2.putText(frame, f"Progress: {progress:.1f}% | Frame: {frame_count}", 
                       (10, frame.shape[0] - 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 100), 2)
        
        cv2.imshow('Video Dugaar Tanikh - Strict Mode', frame)
        
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord(' '):
            paused = not paused
            print("⏸️  Зогсоосон" if paused else "▶️  Үргэлжилж байна")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Дүгнэлт
    print("\n" + "="*70)
    print(" "*25 + "📊 ДҮГНЭЛТ")
    print("="*70)
    print(f"Нийт БОДИТ дугаар: {len(detector.detected_plates)}")
    
    if detector.detected_plates:
        avg_conf = sum(d['confidence'] for d in detector.detected_plates) / len(detector.detected_plates)
        print(f"Дундаж итгэл: {avg_conf:.1f}%")
        
        print(f"\n📋 Бүх дугаарууд:")
        for i, det in enumerate(detector.detected_plates, 1):
            print(f"  {i}. {det['plate']} ({det['confidence']:.1f}%) - {det['time'].strftime('%H:%M:%S')}")
    
    print(f"\n💾 Файлууд: {detector.save_folder}/")
    print("\n👋 Баяртай!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()