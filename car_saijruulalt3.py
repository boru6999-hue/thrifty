import cv2
import numpy as np
import pytesseract
from datetime import datetime
import os
from collections import deque

# Tesseract зам
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class AutoPlateDetector:
    def __init__(self):
        # Хавтас үүсгэх
        self.save_folder = "detected_plates"
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        
        # Дугаар илрүүлэх classifier
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
        )
        
        # Статистик
        self.total_detected = 0
        self.successful_reads = 0
        
        # Сүүлд таньсан дугаарууд (давхцах үед таньхгүй байх)
        self.recent_plates = deque(maxlen=10)
        
        # Одоо таньж буй дугаарууд
        self.current_detections = []
        
        print("✅ Автомат таних систем бэлэн!")
    
    def is_duplicate(self, text, threshold=0.8):
        """Давхцсан дугаар эсэхийг шалгах"""
        if not text or text == "Unknown":
            return False
            
        for recent in self.recent_plates:
            if recent == text:
                return True
            # Ижил төстэй эсэхийг шалгах
            similarity = sum(a == b for a, b in zip(text, recent)) / max(len(text), len(recent))
            if similarity > threshold:
                return True
        return False
    
    def detect_plates(self, frame):
        """Дугаар илрүүлэх"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plates = self.cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 20))
        return plates
    
    def preprocess_plate(self, plate_img):
        """Зураг сайжруулах"""
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Томруулах
        h, w = gray.shape
        if h < 100:
            scale = 100 / h
            new_w = int(w * scale)
            gray = cv2.resize(gray, (new_w, 100))
        
        # Noise арилгах
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # OTSU threshold
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def recognize_text(self, plate_img):
        """OCR - хэд хэдэн аргаар туршина"""
        try:
            processed = self.preprocess_plate(plate_img)
            
            configs = [
                '--oem 3 --psm 7',
                '--oem 3 --psm 8',
                '--oem 3 --psm 11'
            ]
            
            results = []
            for config in configs:
                try:
                    text = pytesseract.image_to_string(processed, config=config)
                    cleaned = self.clean_text(text)
                    if cleaned and len(cleaned) >= 3:
                        results.append(cleaned)
                except:
                    pass
            
            if results:
                from collections import Counter
                counter = Counter(results)
                best_text = counter.most_common(1)[0][0]
                return best_text
        except:
            pass
        
        return None
    
    def clean_text(self, text):
        """Текст цэвэрлэх"""
        cleaned = ''.join(c for c in text if c.isalnum())
        return cleaned.upper().strip()
    
    def draw_detection_box(self, frame, x, y, w, h, text, is_new):
        """Дугаар дээр хүрээ зурах"""
        # Өнгө: шинэ бол ногоон, хуучин бол шар
        color = (0, 255, 0) if is_new else (0, 200, 255)
        
        # Хүрээ
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # Дугаарын текст (том, хар дэвсгэр дээр)
        if text and text != "Unknown":
            # Дэвсгэр
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)[0]
            bg_x1 = x
            bg_y1 = y - 45
            bg_x2 = x + text_size[0] + 20
            bg_y2 = y - 5
            
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 2)
            
            # Текст
            cv2.putText(frame, text, (x + 10, y - 15), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
        
        return frame
    
    def create_detection_window(self, plate_img, text):
        """Том харуулах цонх үүсгэх"""
        h, w = plate_img.shape[:2]
        
        # Том болгох (400px өндөр)
        scale = 200 / h
        new_w = int(w * scale)
        new_h = 200
        enlarged = cv2.resize(plate_img, (new_w, new_h))
        
        # Цонхны хэмжээ
        window_h = new_h + 120
        window_w = max(new_w, 400)
        
        # Хар дэвсгэр үүсгэх
        window = np.zeros((window_h, window_w, 3), dtype=np.uint8)
        window[:] = (30, 30, 30)
        
        # Зураг тавих
        x_offset = (window_w - new_w) // 2
        window[10:10+new_h, x_offset:x_offset+new_w] = enlarged
        
        # Дугаар бичих
        text_y = new_h + 40
        if text and text != "Unknown":
            cv2.putText(window, text, (20, text_y), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3)
        else:
            cv2.putText(window, "Tanigdsan gui", (20, text_y), 
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 100, 255), 2)
        
        # Огноо
        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(window, time_now, (20, text_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
        
        return window
    
    def save_detection(self, plate_img, text):
        """Дугаар хадгалах"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        img_file = os.path.join(self.save_folder, f"plate_{timestamp}.jpg")
        cv2.imwrite(img_file, plate_img)
        
        txt_file = os.path.join(self.save_folder, f"plate_{timestamp}.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"Огноо: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Дугаар: {text if text else 'Танигдсангүй'}\n")
        
        return img_file
    
    def draw_ui(self, frame):
        """UI элементүүд"""
        h, w = frame.shape[:2]
        
        # Дээд хэсэг - статус
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, "AVTOMAAT DUGAAR TANIKH", 
                   (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2)
        
        stats = f"Niit: {self.total_detected} | Amjilttai: {self.successful_reads}"
        cv2.putText(frame, stats, 
                   (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
        
        # Q товч заавар
        cv2.putText(frame, "Q - Garah", 
                   (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

def main():
    print("\n" + "="*70)
    print(" "*15 + "🚗 АВТОМАТ МАШИНЫ ДУГААР ТАНИХ 🚗")
    print("="*70)
    print("\n📌 Онцлог:")
    print("  ✓ Камераас шууд таних")
    print("  ✓ Машин гарч ирэх үед автоматаар таних")
    print("  ✓ Дугаарыг том харуулах")
    print("  ✓ Автоматаар хадгалах")
    print("  ✓ Давхцсан дугаар алгасах")
    print("\n" + "-"*70 + "\n")
    
    # Камер эхлүүлэх
    print("🎥 Камер холбогдож байна...")
    cap = None
    for i in [0, 1, 2]:
        test_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if test_cap.isOpened():
            ret, _ = test_cap.read()
            if ret:
                cap = test_cap
                print(f"✅ Камер #{i} холбогдлоо!\n")
                break
            test_cap.release()
    
    if not cap:
        print("❌ Камер олдсонгүй!")
        return
    
    # Resolution тохируулах
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    detector = AutoPlateDetector()
    
    print("🚀 СИСТЕМИЙГ ЭХЛҮҮЛЖ БАЙНА...")
    print("   Машины дугаарыг камер руу харуулаарай")
    print("   Автоматаар таниж, том харуулна")
    print("   'Q' товч дарж гарах\n")
    print("-"*70 + "\n")
    
    frame_count = 0
    detection_windows = {}  # Хадгалсан цонхнууд
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Камерыг эргүүлэх (хэрэв шаардлагатай бол)
        frame = cv2.flip(frame, 1)
        
        # 3 frame тутамд л таних (хурд нэмэх)
        if frame_count % 3 == 0:
            plates = detector.detect_plates(frame)
            
            for (x, y, w, h) in plates:
                # Дугаарын хэсэг таслах
                plate_img = frame[y:y+h, x:x+w]
                
                # OCR хийх
                text = detector.recognize_text(plate_img)
                
                # Шинэ дугаар эсэхийг шалгах
                is_new = False
                if text and not detector.is_duplicate(text):
                    is_new = True
                    detector.total_detected += 1
                    detector.recent_plates.append(text)
                    
                    if text != "Unknown":
                        detector.successful_reads += 1
                        
                        # Хадгалах
                        detector.save_detection(plate_img, text)
                        
                        # Консолд хэвлэх
                        print(f"✅ Шинэ дугаар: {text} | Нийт: {detector.total_detected}")
                        
                        # Том харуулах цонх үүсгэх
                        window = detector.create_detection_window(plate_img, text)
                        window_name = f"Dugaar - {text}"
                        
                        # Цонх харуулах
                        cv2.imshow(window_name, window)
                        detection_windows[window_name] = True
                
                # Дугаар дээр хүрээ зурах
                frame = detector.draw_detection_box(frame, x, y, w, h, text, is_new)
        
        # UI зурах
        frame = detector.draw_ui(frame)
        
        # Үндсэн цонх
        cv2.imshow('Avtomaat Tanikh Sistem', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
    
    # Дуусгавар
    cap.release()
    cv2.destroyAllWindows()
    
    # Дүгнэлт
    print("\n" + "="*70)
    print(" "*25 + "📊 ДҮГНЭЛТ")
    print("="*70)
    print(f"Нийт илрүүлсэн: {detector.total_detected}")
    print(f"Амжилттай уншсан: {detector.successful_reads}")
    if detector.total_detected > 0:
        accuracy = (detector.successful_reads / detector.total_detected) * 100
        print(f"Нарийвчлал: {accuracy:.1f}%")
    print(f"\n💾 Файлууд хадгалагдсан: {detector.save_folder}/")
    print("\n👋 Баяртай!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()