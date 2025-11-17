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
        self.detected_plates = []
        self.recent_plates = deque(maxlen=10)  # 5 -> 10 болгов
        
        # Монгол дугаарын форматууд
        self.plate_patterns = [
            r'^[0-9]{4}[A-Z]{3}$',      # 1234ABC
            r'^[A-Z]{3}[0-9]{4}$',      # ABC1234
            r'^[0-9]{3}[A-Z]{2}[0-9]{2}$',  # 123AB12
            r'^[A-Z]{2}[0-9]{4}$',      # AB1234
            r'^[0-9]{2}[A-Z]{2}[0-9]{3}$',  # 12AB123
        ]
        
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
        if not text or len(text) < 4:
            return True
        # Төстэй дугаар хайх (1 тэмдэгт зөрүүтэй ч)
        for plate in self.recent_plates:
            similarity = sum(a == b for a, b in zip(text, plate))
            if len(text) == len(plate) and similarity >= len(text) - 1:
                return True
        return False
    
    def is_valid_mongolian_plate(self, text):
        """Монгол дугаарын формат эсэхийг шалгах"""
        if not text or len(text) < 5 or len(text) > 8:
            return False
        
        for pattern in self.plate_patterns:
            if re.match(pattern, text):
                return True
        return False
    
    def detect_plates(self, frame):
        """Дугаар илрүүлэх - ИЛҮҮ НАРИЙВЧЛАЛТАЙ"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Contrast сайжруулах
        gray = cv2.equalizeHist(gray)
        
        # Шум арилгах
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Олон scale-ээр хайх
        all_plates = []
        for scale in [1.03, 1.05, 1.08, 1.1]:
            plates = self.cascade.detectMultiScale(
                gray, 
                scaleFactor=scale,
                minNeighbors=3,
                minSize=(80, 25),  # Бага дугаар
                maxSize=(400, 150)
            )
            if len(plates) > 0:
                all_plates.extend(plates)
        
        # Давхцсан хүрээнүүдийг арилгах
        if len(all_plates) > 0:
            all_plates = self.remove_overlapping(all_plates)
        
        return all_plates
    
    def remove_overlapping(self, boxes):
        """Давхцсан бокс арилгах"""
        if len(boxes) == 0:
            return []
        
        boxes = np.array(boxes)
        pick = []
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        area = boxes[:, 2] * boxes[:, 3]
        
        idxs = np.argsort(area)[::-1]  # Том box эхэлж
        
        while len(idxs) > 0:
            i = idxs[0]
            pick.append(i)
            
            # Overlap шалгах
            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            overlap = (w * h) / area[idxs[1:]]
            
            idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > 0.3)[0] + 1)))
        
        return boxes[pick]
    
    def preprocess_plate(self, plate_img):
        """Зураг МААШГҮЙ САЙН боловсруулах"""
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()
        
        # 1. ТОМ БОЛГОХ (300 pixel өндөр)
        h, w = gray.shape
        if h < 100:  # Хэт бага бол
            scale = 300 / h
            new_w = int(w * scale)
            gray = cv2.resize(gray, (new_w, 300), interpolation=cv2.INTER_CUBIC)
        
        # 2. Шум арилгах - ХҮЧТЭЙ
        gray = cv2.fastNlMeansDenoising(gray, h=15)
        
        # 3. Contrast - CLAHE
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # 4. Гэрэлтүүлэг тэгшлэх
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # 5. Sharpening - ХҮЧТЭЙ
        kernel = np.array([[-1,-1,-1],
                          [-1, 10,-1],
                          [-1,-1,-1]])
        gray = cv2.filter2D(gray, -1, kernel)
        
        # 6. Morphology - текст тодруулах
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # 7. OTSU threshold (adaptive-аас дээр)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 8. Дахин цэвэрлэх
        binary = cv2.medianBlur(binary, 3)
        
        # 9. Dilate - тэмдэгтүүд холбох
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        binary = cv2.dilate(binary, kernel, iterations=1)
        
        return binary
    
    def recognize_text(self, plate_img):
        """OCR - ӨНДӨР НАРИЙВЧЛАЛТАЙ"""
        try:
            processed = self.preprocess_plate(plate_img)
            
            # OCR тохиргоо - Монгол дугаарт тохируулсан
            configs = [
                '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                '--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            ]
            
            results = []
            for config in configs:
                try:
                    # Эх зураг
                    text1 = pytesseract.image_to_string(processed, config=config)
                    cleaned1 = self.clean_text(text1)
                    if cleaned1 and self.is_valid_mongolian_plate(cleaned1):
                        results.append(cleaned1)
                    
                    # Урвуу зураг
                    inverted = cv2.bitwise_not(processed)
                    text2 = pytesseract.image_to_string(inverted, config=config)
                    cleaned2 = self.clean_text(text2)
                    if cleaned2 and self.is_valid_mongolian_plate(cleaned2):
                        results.append(cleaned2)
                except:
                    pass
            
            if results:
                from collections import Counter
                counter = Counter(results)
                best_text, count = counter.most_common(1)[0]
                confidence = (count / len(results)) * 100 if results else 0
                
                # Зөвхөн өндөр итгэлттэй үр дүн буцаах
                if confidence >= 30:  # 30%-аас дээш
                    return best_text, confidence
        except Exception as e:
            pass
        
        return None, 0
    
    def clean_text(self, text):
        """Текст цэвэрлэх + ЗАСАХ"""
        cleaned = ''.join(c for c in text if c.isalnum())
        cleaned = cleaned.upper().strip()
        
        # Алдаатай тэмдэгт засах
        replacements = {
            'O': '0', 'I': '1', 'Z': '2', 'S': '5', 
            'B': '8', 'G': '6', 'Q': '0', 'D': '0',
            '0': 'O', '1': 'I', '5': 'S', '8': 'B',  # Эсрэг чиглэл
        }
        
        # Тоо ба үсгийн байршил харгалзан засах
        result = []
        for i, char in enumerate(cleaned):
            # Эхний 2-4 тэмдэгт тоо байх ёстой (ихэнх тохиолдолд)
            if i < 3 and char.isalpha() and char in replacements.values():
                # Үсгийг тоо болгох
                for k, v in replacements.items():
                    if v == char and k.isdigit():
                        char = k
                        break
            # Сүүлийн 2-3 тэмдэгт үсэг байх ёстой
            elif i >= len(cleaned) - 3 and char.isdigit() and char in replacements:
                char = replacements[char]
            
            result.append(char)
        
        cleaned = ''.join(result)
        
        if len(cleaned) < 5 or len(cleaned) > 8:
            return None
        
        return cleaned
    
    def draw_table(self, frame):
        """Хүснэгт зурах"""
        h, w = frame.shape[:2]
        table_width = 350
        table_x = w - table_width
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (table_x, 0), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        cv2.rectangle(frame, (table_x, 0), (w, 50), (0, 100, 200), -1)
        cv2.putText(frame, "TANISAN DUGAARUD", 
                   (table_x + 10, 32), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        
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
        
        start_idx = max(0, len(self.detected_plates) - 12)
        for i, detection in enumerate(self.detected_plates[start_idx:]):
            y += 40
            if y > h - 20:
                break
            
            time_str = detection['time'].strftime("%H:%M:%S")
            plate = detection['plate']
            conf = detection['confidence']
            
            cv2.putText(frame, time_str, (table_x + 10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            color = (0, 255, 0) if conf > 70 else (0, 200, 255) if conf > 40 else (0, 100, 255)
            cv2.putText(frame, plate, (table_x + 100, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            
            cv2.putText(frame, f"{conf:.0f}", (table_x + 280, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.rectangle(frame, (table_x, h - 60), (w, h), (40, 40, 40), -1)
        cv2.putText(frame, f"Niit: {len(self.detected_plates)}", 
                   (table_x + 10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 1)
        
        high_conf = sum(1 for d in self.detected_plates if d['confidence'] > 70)
        cv2.putText(frame, f"Ondor: {high_conf}", 
                   (table_x + 10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 1)
        
        return frame
    
    def draw_detection(self, frame, x, y, w, h, text, confidence):
        """Дугаар дээр хүрээ"""
        if confidence > 70:
            color = (0, 255, 0)
        elif confidence > 40:
            color = (0, 200, 255)
        else:
            color = (0, 100, 255)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
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
    print(" "*10 + "🚗 САЙЖРУУЛСАН ВИДЕО ДУГААР ТАНИХ 🚗")
    print("="*70)
    print("\n📌 Шинэ онцлог:")
    print("  ✓ Давхцсан box арилгах")
    print("  ✓ Монгол дугаарын формат шалгах")
    print("  ✓ Алдаатай тэмдэгт засах (O→0, I→1)")
    print("  ✓ Зураг илүү том болгох (300px)")
    print("  ✓ Олон scale-ээр хайх")
    print("  ✓ Confidence шалгуур хатуу болгосон")
    print("\n" + "-"*70 + "\n")
    
    detector = VideoPlateDetector()
    
    print("📁 Видео файл сонгох цонх нээгдэнэ...")
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
            
            # Resolution - БАГА БАГАСГАХ (1920px хүртэл)
            h, w = frame.shape[:2]
            if w > 1920:
                scale = 1920 / w
                frame = cv2.resize(frame, (1920, int(h * scale)))
            
            # 3 frame тутамд таних (5→3 болгов)
            if frame_count % 3 == 0:
                plates = detector.detect_plates(frame)
                
                for (x, y, w_p, h_p) in plates:
                    # Зураг огтолох үед margin нэмэх
                    margin = 5
                    y1 = max(0, y - margin)
                    y2 = min(frame.shape[0], y + h_p + margin)
                    x1 = max(0, x - margin)
                    x2 = min(frame.shape[1], x + w_p + margin)
                    
                    plate_img = frame[y1:y2, x1:x2]
                    
                    if plate_img.size == 0:
                        continue
                    
                    text, confidence = detector.recognize_text(plate_img)
                    
                    if text and not detector.is_duplicate(text):
                        detector.detected_plates.append({
                            'time': datetime.now(),
                            'plate': text,
                            'confidence': confidence,
                            'image': plate_img.copy()
                        })
                        detector.recent_plates.append(text)
                        
                        print(f"✅ Шинэ: {text} ({confidence:.0f}%) | Нийт: {len(detector.detected_plates)}")
                    
                    if text:
                        frame = detector.draw_detection(frame, x, y, w_p, h_p, text, confidence)
            
            frame = detector.draw_table(frame)
            
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
    
    if detector.detected_plates:
        detector.save_to_file()
    
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