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

class FastPlateDetector:
    def __init__(self):
        self.save_folder = "detected_plates"
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
        )
        
        self.detected_plates = []
        
        # ХАТУУ давхцал хяналт
        self.detected_texts = set()  # Нэг удаа л таних
        
        print("✅ Хурдан систем бэлэн!")
    
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
        """Дугаарын формат шалгах - ХАТУУ"""
        if not text or len(text) < 5 or len(text) > 10:
            return False
        
        # ЗААВАЛ тоо байх ёстой (2+)
        digit_count = sum(c.isdigit() for c in text)
        if digit_count < 2:
            return False
        
        # ЗААВАЛ үсэг байх ёстой
        letter_count = sum(c.isalpha() for c in text)
        if letter_count < 1:
            return False
        
        # Зөвхөн үсэг тоо
        if not text.isalnum():
            return False
        
        # Эхний тэмдэгт тоо БИش байх (ихэнх дугаар үсэгээр эхэлнэ)
        # Жишээ: УБ1234 ✓, 1234УБ ✗
        if text[0].isdigit():
            # Хэрэв эхлээд тоо байвал, ядаж 3 үсэг дараа нь байх ёстой
            if letter_count < 3:
                return False
        
        return True
    
    def detect_plates(self, frame):
        """ХУРДАН илрүүлэлт"""
        # Багасгаад илрүүлэх (хурдан)
        small = cv2.resize(frame, (640, 360))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        plates = self.cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(40, 15)
        )
        
        # Координатыг эх хэмжээнд буцаах
        scale_x = frame.shape[1] / 640
        scale_y = frame.shape[0] / 360
        
        scaled_plates = []
        for (x, y, w, h) in plates:
            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)
            
            # Харьцаа шалгах
            ratio = w / h
            if 2.0 <= ratio <= 5.5:
                scaled_plates.append((x, y, w, h))
        
        return scaled_plates
    
    def enhance_plate_fast(self, plate_img):
        """ХУРДАН боловсруулалт"""
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()
        
        h, w = gray.shape
        
        # Томруулах - БАГААР (хурдан)
        target_h = 120  # Багасгасан
        scale = target_h / h
        new_w = int(w * scale)
        gray = cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # Энгийн denoise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # 2 төрлийн threshold
        _, binary1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
        
        return [binary1, binary2]
    
    def ocr_robust(self, images):
        """ҮР ДҮНТЭЙ OCR - Монгол дугаарт тохирсон"""
        all_results = []
        
        # Монгол болон англи үсэг
        whitelist = '0123456789АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
        
        # Олон config туршина
        configs = [
            f'--oem 3 --psm 7 -c tessedit_char_whitelist={whitelist}',
            f'--oem 3 --psm 8 -c tessedit_char_whitelist={whitelist}',
            f'--oem 1 --psm 7 -c tessedit_char_whitelist={whitelist}',
        ]
        
        for img in images:
            for config in configs:
                try:
                    # OCR data авах (confidence-тай)
                    data = pytesseract.image_to_data(img, config=config, 
                                                     output_type=pytesseract.Output.DICT)
                    
                    # Итгэлтэй текстүүд авах
                    texts = []
                    confs = []
                    
                    for i in range(len(data['text'])):
                        conf = int(data['conf'][i])
                        text = data['text'][i].strip()
                        
                        if conf > 50 and len(text) > 0:  # 50%+ итгэл
                            cleaned = self.clean_text(text)
                            if cleaned and len(cleaned) >= 2:
                                texts.append(cleaned)
                                confs.append(conf)
                    
                    # Нэгтгэх
                    if texts:
                        full_text = ''.join(texts)
                        avg_conf = sum(confs) / len(confs)
                        
                        if self.is_valid_plate(full_text):
                            all_results.append((full_text, avg_conf))
                
                except:
                    pass
        
        # Хамгийн сайн үр дүн
        if all_results:
            # Confidence-аар эрэмбэлэх
            all_results.sort(key=lambda x: x[1], reverse=True)
            
            # Хамгийн өндөр confidence
            best_text, best_conf = all_results[0]
            
            # Шалгуур: 65%+ байх ёстой
            if best_conf >= 65:
                return best_text, best_conf
        
        return None, 0
    
    def clean_text(self, text):
        """Текст цэвэрлэх + засварлах"""
        # Зөвхөн үсэг тоо
        cleaned = ''.join(c for c in text if c.isalnum())
        cleaned = cleaned.upper().strip()
        
        # Түгээмэл алдаа засах
        replacements = {
            'O': '0',  # O -> 0
            'I': '1',  # I -> 1
            'Z': '2',  # Z -> 2
            'S': '5',  # S -> 5
            'B': '8',  # B -> 8
            'G': '6',  # G -> 6 (танай асуудал!)
            'D': '0',  # D -> 0
            'Q': '0',  # Q -> 0
        }
        
        # Зөвхөн тоо байх ёстой хэсэгт засварлах
        # Монгол дугаар: обычно 2-4 үсэг + 4 тоо
        result = []
        for i, char in enumerate(cleaned):
            # Хэрэв 3+ дахь тэмдэгт бол тоо байх магадлал өндөр
            if i >= 3 and char in replacements:
                result.append(replacements[char])
            else:
                result.append(char)
        
        cleaned = ''.join(result)
        
        if len(cleaned) < 5 or len(cleaned) > 10:
            return None
        
        return cleaned
    
    def format_video_time(self, frame_number, fps):
        """Видеоны цаг формат"""
        total_seconds = int(frame_number / fps)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"
    
    def draw_table(self, frame, fps, current_frame):
        """Хүснэгт"""
        h, w = frame.shape[:2]
        table_w = 400
        table_x = w - table_w
        
        # Дэвсгэр
        cv2.rectangle(frame, (table_x, 0), (w, h), (15, 15, 15), -1)
        
        # Гарчиг
        cv2.rectangle(frame, (table_x, 0), (w, 60), (0, 120, 0), -1)
        cv2.putText(frame, "TANISAN DUGAARUD", 
                   (table_x + 60, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        
        # Толгой
        y = 80
        cv2.line(frame, (table_x + 10, y), (w - 10, y), (80, 80, 80), 2)
        y += 30
        
        cv2.putText(frame, "#", (table_x + 15, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "Tsag", (table_x + 50, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "Dugaar", (table_x + 140, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, "%", (table_x + 320, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        y += 10
        cv2.line(frame, (table_x + 10, y), (w - 10, y), (80, 80, 80), 1)
        
        # Мөрүүд
        for i, det in enumerate(self.detected_plates[-9:], 1):
            y += 40
            if y > h - 100:
                break
            
            video_time = det['video_time']
            plate = det['plate']
            conf = det['confidence']
            
            color = (0, 255, 0) if conf >= 80 else (0, 220, 220)
            
            cv2.putText(frame, f"{len(self.detected_plates) - 9 + i}", 
                       (table_x + 15, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            cv2.putText(frame, video_time, (table_x + 50, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.putText(frame, plate, (table_x + 140, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.putText(frame, f"{conf:.0f}", (table_x + 320, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # Статистик
        cv2.rectangle(frame, (table_x, h-70), (w, h), (25, 25, 25), -1)
        
        cv2.putText(frame, f"Niit olson: {len(self.detected_plates)}", 
                   (table_x + 15, h - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
        
        if self.detected_plates:
            avg = sum(d['confidence'] for d in self.detected_plates) / len(self.detected_plates)
            cv2.putText(frame, f"Dundaj: {avg:.1f}%", 
                       (table_x + 15, h - 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
        
        # Одоогийн видео цаг
        current_time = self.format_video_time(current_frame, fps)
        cv2.putText(frame, f"Video: {current_time}", 
                   (table_x + 250, h - 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return frame
    
    def draw_detection(self, frame, x, y, w, h, text, conf):
        """Дугаар дээр хүрээ"""
        color = (0, 255, 0) if conf >= 80 else (0, 220, 220)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        if text:
            # Дэвсгэр
            label = f"{text}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
            
            cv2.rectangle(frame, (x, y-35), (x + text_size[0] + 15, y), (0, 0, 0), -1)
            cv2.rectangle(frame, (x, y-35), (x + text_size[0] + 15, y), color, 2)
            
            cv2.putText(frame, label, (x + 7, y - 10), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
        
        return frame
    
    def create_result_window(self, plate_img, text, conf, video_time):
        """Үр дүн цонх"""
        h, w = plate_img.shape[:2]
        scale = 180 / h
        enlarged = cv2.resize(plate_img, (int(w*scale), 180))
        
        window_h = 320
        window_w = max(enlarged.shape[1] + 40, 450)
        window = np.zeros((window_h, window_w, 3), dtype=np.uint8)
        window[:] = (20, 20, 20)
        
        # Гарчиг
        cv2.rectangle(window, (0, 0), (window_w, 50), (0, 100, 200), -1)
        cv2.putText(window, "TANISAN DUGAAR", 
                   (window_w//2 - 130, 35), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2)
        
        # Зураг
        x_off = (window_w - enlarged.shape[1]) // 2
        window[60:240, x_off:x_off+enlarged.shape[1]] = enlarged
        
        # Дугаар - ТОМ
        color = (0, 255, 0) if conf >= 80 else (0, 220, 220)
        cv2.putText(window, text, (20, 280), 
                   cv2.FONT_HERSHEY_DUPLEX, 1.8, color, 3)
        
        # Мэдээлэл
        info = f"{conf:.0f}% | Video: {video_time}"
        cv2.putText(window, info, (20, 310), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        return window
    
    def save_result(self, plate_img, text, conf, video_time):
        """Хадгалах"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Зураг
        img_file = os.path.join(self.save_folder, f"{text}_{video_time.replace(':', '-')}_{timestamp}.jpg")
        cv2.imwrite(img_file, plate_img)
        
        # Текст
        txt_file = os.path.join(self.save_folder, f"log.txt")
        with open(txt_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Video: {video_time} | {text} | {conf:.1f}%\n")

def main():
    print("\n" + "="*70)
    print(" "*10 + "🚗 ВИДЕО ДУГААР ТАНИХ (ЭЦСИЙН ХУВИЛБАР) 🚗")
    print("="*70)
    print("\n⚡ Онцлог:")
    print("  • ХУРДАН (багасгаад илрүүлнэ)")
    print("  • НЭГ удаа л таних (давхцахгүй)")
    print("  • Видеоны цаг харуулна (00:15)")
    print("  • Алдаа засна (G→6, O→0, I→1)")
    print("  • Монгол дугаар дэмжинэ")
    print("\n" + "-"*70 + "\n")
    
    detector = FastPlateDetector()
    
    print("📁 Видео сонгох...")
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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps
    
    print(f"📊 Мэдээлэл:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Duration: {int(duration//60)}:{int(duration%60):02d}")
    print(f"   Frames: {total_frames}\n")
    
    # Display size
    display_w = 1280 if width > 1280 else width
    display_h = int(height * (display_w / width))
    
    print("🚀 Эхэлж байна...\n")
    print("   SPACE - Зогсоох/Үргэлжлүүлэх")
    print("   Q - Дуусгах\n")
    print("-"*70 + "\n")
    
    frame_count = 0
    paused = False
    
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
            
            # Resize
            if frame.shape[1] != display_w:
                frame = cv2.resize(frame, (display_w, display_h))
            
            # FPS
            curr_time = cv2.getTickCount()
            time_diff = (curr_time - prev_time) / cv2.getTickFrequency()
            if time_diff > 0:
                fps_display = 1.0 / time_diff
            prev_time = curr_time
            
            # 8 frame тутамд таних (ХУРДАН)
            if frame_count % 8 == 0:
                plates = detector.detect_plates(frame)
                
                for (x, y, w, h) in plates:
                    plate_img = frame[y:y+h, x:x+w]
                    
                    # Боловсруулах
                    enhanced = detector.enhance_plate_fast(plate_img)
                    
                    # OCR
                    text, conf = detector.ocr_robust(enhanced)
                    
                    # Шалгах: өндөр итгэл + ДАВХЦААГҮЙ
                    if text and conf >= 65:
                        if text not in detector.detected_texts:
                            # НЭГ удаа л таних!
                            detector.detected_texts.add(text)
                            
                            video_time = detector.format_video_time(frame_count, fps)
                            
                            detector.detected_plates.append({
                                'video_time': video_time,
                                'plate': text,
                                'confidence': conf
                            })
                            
                            detector.save_result(plate_img, text, conf, video_time)
                            
                            print(f"✅ {len(detector.detected_plates)}. {text} ({conf:.0f}%) - {video_time}")
                            
                            # Үр дүн цонх
                            result_win = detector.create_result_window(
                                plate_img, text, conf, video_time
                            )
                            cv2.imshow(f"{text} - {video_time}", result_win)
                    
                    # Зурах
                    if text and conf >= 65:
                        frame = detector.draw_detection(frame, x, y, w, h, text, conf)
            
            # Хүснэгт
            frame = detector.draw_table(frame, fps, frame_count)
            
            # Статус
            h_frame = frame.shape[0]
            cv2.rectangle(frame, (5, h_frame-60), (350, h_frame-5), (25, 25, 25), -1)
            
            cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, h_frame-35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
            
            progress = (frame_count / total_frames) * 100
            cv2.putText(frame, f"Progress: {progress:.1f}%", (10, h_frame-12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        
        cv2.imshow('Video Plate Detection - Final', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("\n🛑 Зогсоосон")
            break
        elif key == ord(' '):
            paused = not paused
            print("⏸️  ЗОГСООСОН" if paused else "▶️  ҮРГЭЛЖИЛЖ БАЙНА")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Тайлан
    print("\n" + "="*70)
    print(" "*25 + "📊 ДҮГНЭЛТ")
    print("="*70)
    print(f"Танигдсан дугаар: {len(detector.detected_plates)}")
    
    if detector.detected_plates:
        print(f"\n📋 Бүх дугаарууд:")
        for i, det in enumerate(detector.detected_plates, 1):
            print(f"  {i}. {det['plate']} ({det['confidence']:.0f}%) - Video: {det['video_time']}")
        
        avg = sum(d['confidence'] for d in detector.detected_plates) / len(detector.detected_plates)
        print(f"\nДундаж confidence: {avg:.1f}%")
    
    print(f"\n💾 Хадгалагдсан: {detector.save_folder}/")
    print("   - Зургууд")
    print("   - log.txt файл")
    print("\n👋 Баяртай!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()