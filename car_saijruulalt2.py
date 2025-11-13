import cv2
import numpy as np
import pytesseract
from datetime import datetime
import os

# Tesseract зам
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class SimplePlateDetector:
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
        self.total_scans = 0
        self.successful_scans = 0
        
        print("✅ Систем бэлэн болсон!")
    
    def detect_plates(self, frame):
        """Дугаар илрүүлэх"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plates = self.cascade.detectMultiScale(gray, 1.1, 4)
        return plates
    
    def preprocess_plate(self, plate_img):
        """Зураг сайжруулах - энгийн боловч үр дүнтэй"""
        # Саарал болгох
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Жижиг бол томруулах
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
        processed = self.preprocess_plate(plate_img)
        
        # 3 өөр config туршина
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
        
        # Хамгийн их давтагдсан текст
        if results:
            # Давтамж тоолох
            from collections import Counter
            counter = Counter(results)
            best_text = counter.most_common(1)[0][0]
            return best_text
        
        return None
    
    def clean_text(self, text):
        """Текст цэвэрлэх"""
        # Зөвхөн үсэг тоо
        cleaned = ''.join(c for c in text if c.isalnum())
        return cleaned.upper().strip()
    
    def draw_ui(self, frame, plates):
        """Энгийн гоё UI"""
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Дээд хэсэг - Нэр
        cv2.rectangle(display, (0, 0), (w, 80), (50, 50, 50), -1)
        cv2.putText(display, "MASHINII DUGAAR TANIKH SISTEM", 
                   (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(display, "SPACE - Tanikh | Q - Garah", 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Статистик
        stats_text = f"Niit: {self.total_scans} | Amjilttai: {self.successful_scans}"
        cv2.putText(display, stats_text, 
                   (w - 300, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Дугаарууд дээр хүрээ зурах
        for i, (x, y, w_plate, h_plate) in enumerate(plates):
            # Ногоон хүрээ
            cv2.rectangle(display, (x, y), (x+w_plate, y+h_plate), (0, 255, 0), 3)
            
            # Дугуй дугаар
            cv2.circle(display, (x-15, y+h_plate//2), 15, (0, 255, 0), -1)
            cv2.putText(display, str(i+1), (x-20, y+h_plate//2+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Label
            cv2.putText(display, f"Dugaar #{i+1}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Доод хэсэг - Огноо
        cv2.rectangle(display, (0, h-35), (w, h), (50, 50, 50), -1)
        time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(display, time_now, (20, h-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        if plates is not None and len(plates) > 0:
            cv2.putText(display, f"{len(plates)} dugaar oldloo!", 
                       (w-200, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        return display
    
    def save_result(self, plate_img, text):
        """Үр дүн хадгалах"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Зураг хадгалах
        img_file = os.path.join(self.save_folder, f"plate_{timestamp}.jpg")
        cv2.imwrite(img_file, plate_img)
        
        # Текст файлд хадгалах (энгийн)
        txt_file = os.path.join(self.save_folder, f"plate_{timestamp}.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"Огноо: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Дугаар: {text if text else 'Танигдсангүй'}\n")
        
        return img_file

def main():
    print("\n" + "="*60)
    print(" "*15 + "🚗 МАШИНЫ ДУГААР ТАНИХ 🚗")
    print("="*60)
    print("\n📌 Онцлог:")
    print("  ✓ Камераас дүрс авах")
    print("  ✓ Дугаар автомат илрүүлэх")
    print("  ✓ OCR ашиглан текст таних")
    print("  ✓ Зураг болон текст хадгалах")
    print("  ✓ Гоё харагдалт")
    print("\n" + "-"*60 + "\n")
    
    # Камер олох
    print("🎥 Камер хайж байна...")
    cap = None
    for i in [0, 1, 2]:
        test_cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if test_cap.isOpened():
            ret, _ = test_cap.read()
            if ret:
                cap = test_cap
                print(f"✅ Камер #{i} амжилттай холбогдлоо!\n")
                break
            test_cap.release()
    
    if not cap:
        print("❌ Камер олдсонгүй!")
        return
    
    detector = SimplePlateDetector()
    
    print("⌨️  ТОВЧНУУД:")
    print("  SPACE  - Дугаар таних")
    print("  Q      - Гарах")
    print("-"*60 + "\n")
    print("🚀 Бэлэн! Машины дугаарыг камер руу харуулаарай...\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 🔄 ЗУРГИЙГ ЭРГҮҮЛЭХ (Notebook камерт зориулж)
        frame = cv2.flip(frame, 1)  # Тэнхлэгээр урвуулах
        
        # Хэрэв дээрх нь тааруулаагүй бол доорх аргуудыг туршаарай:
        # frame = cv2.rotate(frame, cv2.ROTATE_180)  # 180° эргүүлэх
        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # 90° баруун
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 90° зүүн
        
        # Дугаар илрүүлэх
        plates = detector.detect_plates(frame)
        
        # UI харуулах
        display = detector.draw_ui(frame, plates)
        
        cv2.imshow('Mashinii Dugaar Tanikh', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        # SPACE - Таних
        if key == ord(' '):
            if len(plates) == 0:
                print("\n⚠️  Дугаар олдсонгүй! Дахин оролдоно уу.\n")
            else:
                print(f"\n{'='*60}")
                print(f"🔍 {len(plates)} дугаар илэрлээ!")
                print('='*60)
                
                for i, (x, y, w, h) in enumerate(plates, 1):
                    detector.total_scans += 1
                    
                    # Дугаарын хэсэг таслах
                    plate_img = frame[y:y+h, x:x+w]
                    
                    print(f"\n📋 Дугаар #{i}:")
                    print("   ⚙️  Боловсруулж байна...")
                    
                    # Таних
                    text = detector.recognize_text(plate_img)
                    
                    if text:
                        print(f"   ✅ Танилт: {text}")
                        detector.successful_scans += 1
                    else:
                        print("   ❌ Танигдсангүй")
                        text = "Unknown"
                    
                    # Хадгалах
                    img_file = detector.save_result(plate_img, text)
                    print(f"   💾 Хадгалсан: {os.path.basename(img_file)}")
                    
                    # Үр дүн харуулах
                    result_img = plate_img.copy()
                    result_h = 150
                    result_w = max(300, result_img.shape[1])
                    
                    # Том харуулах зураг үүсгэх
                    display_result = np.zeros((result_h + result_img.shape[0] + 20, result_w, 3), dtype=np.uint8)
                    display_result[:] = (40, 40, 40)
                    
                    # Зураг тавих
                    y_offset = 10
                    x_offset = (result_w - result_img.shape[1]) // 2
                    display_result[y_offset:y_offset+result_img.shape[0], 
                                 x_offset:x_offset+result_img.shape[1]] = result_img
                    
                    # Текст
                    y_text = y_offset + result_img.shape[0] + 40
                    cv2.putText(display_result, f"Dugaar: {text}", 
                               (20, y_text), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.putText(display_result, "Daraagiin dugaar - SPACE | Garah - ESC", 
                               (20, y_text + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
                    
                    cv2.imshow(f'Ur dun - Dugaar #{i}', display_result)
                
                # Статистик
                print(f"\n📊 Статистик: {detector.successful_scans}/{detector.total_scans} амжилттай")
                print('='*60 + '\n')
                
                # Хүлээх
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                cv2.imshow('Mashinii Dugaar Tanikh', display)
        
        # Q - Гарах
        elif key == ord('q') or key == ord('Q'):
            break
    
    # Дуусгавар
    cap.release()
    cv2.destroyAllWindows()
    
    # Эцсийн мэдээлэл
    print("\n" + "="*60)
    print(" "*20 + "📊 ДҮГНЭЛТ")
    print("="*60)
    print(f"Нийт оролдлого: {detector.total_scans}")
    print(f"Амжилттай танилт: {detector.successful_scans}")
    if detector.total_scans > 0:
        accuracy = (detector.successful_scans / detector.total_scans) * 100
        print(f"Нарийвчлал: {accuracy:.1f}%")
    print(f"Файлууд: {detector.save_folder}/")
    print("\n👋 Баяртай! Амжилт хүсье!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()