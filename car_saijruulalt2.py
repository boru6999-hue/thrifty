import cv2
import numpy as np
import pytesseract
import os
import shutil
import subprocess
import platform
from datetime import datetime, timedelta
from collections import deque
import tkinter as tk
from tkinter import filedialog

# PIL/Pillow for Cyrillic text rendering
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  PIL/Pillow олдсонгүй. Кирилл текст зөв харуулахгүй байж магадгүй.")
    print("   Суулгах: pip install Pillow")

# Tesseract зам
# Tesseract зам (автомат илрүүлэх)
# Default path (Windows installer)
tess_default = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
if os.path.exists(tess_default):
    pytesseract.pytesseract.tesseract_cmd = tess_default
else:
    # Try to find tesseract in PATH
    t_in_path = shutil.which('tesseract')
    if t_in_path:
        pytesseract.pytesseract.tesseract_cmd = t_in_path
    else:
        # Leave default value (may fail later) but print actionable message
        pytesseract.pytesseract.tesseract_cmd = tess_default
        print("⚠️ Tesseract not found. Please install Tesseract OCR for Windows and then run the installer script to add 'mon.traineddata'.")
        print(" - Python installer: run 'python install_mon.py'")
        print(" - PowerShell installer: run 'scripts\\install_mon_traineddata.ps1' as Administrator")
        print(" - Or install Tesseract from: https://github.com/tesseract-ocr/tesseract/releases")


class FastPlateDetector:
    def __init__(self):
        self.save_folder = "detected_plates"
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
        )

        self.detected_plates = []
        self.seen_plates = {}

        # Сүүлд илэрсэн дугаарын зураг хадгалах
        self.last_plate_image = None
        self.last_plate_text = None
        self.last_plate_conf = None
        
        # Хамгийн олон удаа танигдсан дугаар
        self.most_detected_plate = None
        self.most_detected_count = 0

        # Шалгуур
        self.MIN_CONFIDENCE = 50  # Багасгасан (65-аас 50 руу)
        self.MIN_SAME_FRAME_GAP = 60
        self.MIN_SAME_TIME_GAP = 3.0

        # Tesseract хэл шалгах - Монгол эсвэл Орос
        self.ocr_lang = self._detect_ocr_language()
        
        # Debug mode
        self.debug_mode = True
        self.debug_count = 0
        
        # Clickable regions for GUI (plate number -> file path mapping)
        self.clickable_regions = {}  # {(x1, y1, x2, y2): file_path}
        self.plate_to_file = {}  # {plate_text: file_path}
        
        # Одоо байгаа файлуудыг ачаалах
        self._load_existing_files()
        
        print("✅ Хурдан систем бэлэн!")
    
    def put_text_cyrillic(self, frame, text, position, font_scale=0.7, color=(255, 255, 255), thickness=2):
        """Кирилл текстийг зөв харуулах (PIL ашиглах)"""
        if not PIL_AVAILABLE:
            # PIL байхгүй бол энгийн putText ашиглах (ASCII л харуулна)
            try:
                cv2.putText(frame, text.encode('ascii', 'replace').decode('ascii'), 
                           position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            except:
                cv2.putText(frame, "???", position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            return
        
        try:
            x, y = position
            # Текстийн хэмжээг тооцоолох
            font_size = int(font_scale * 40)
            
            # PIL Image үүсгэх
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            
            # Font олох (Windows дээр)
            try:
                # Windows-ийн стандарт font-ууд
                font_paths = [
                    r"C:\Windows\Fonts\arial.ttf",
                    r"C:\Windows\Fonts\calibri.ttf",
                    r"C:\Windows\Fonts\tahoma.ttf",
                ]
                font = None
                for font_path in font_paths:
                    if os.path.exists(font_path):
                        font = ImageFont.truetype(font_path, font_size)
                        break
                
                if font is None:
                    font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            # RGB color
            rgb_color = (color[2], color[1], color[0])
            
            # Текст зурах
            draw.text((x, y), text, font=font, fill=rgb_color)
            
            # OpenCV format руу буцаах
            frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            # Алдаа гарвал энгийн текст ашиглах
            try:
                cv2.putText(frame, text.encode('ascii', 'replace').decode('ascii'), 
                           position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            except:
                cv2.putText(frame, "???", position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    def _load_existing_files(self):
        """Одоо байгаа хадгалагдсан файлуудыг ачаалах (GUI дээр дарахад ашиглах)"""
        try:
            if os.path.exists(self.save_folder):
                for filename in os.listdir(self.save_folder):
                    if filename.endswith('.jpg') and not filename.startswith('_LOW_'):
                        # Файлын нэрээс дугаарыг задлах
                        # Формат: PLATE_TIME_TIMESTAMP.jpg
                        parts = filename.replace('.jpg', '').split('_')
                        if len(parts) >= 1:
                            # Эхний хэсэг нь дугаар байх магадлалтай
                            plate_text = parts[0]
                            if plate_text and len(plate_text) >= 4:
                                file_path = os.path.join(self.save_folder, filename)
                                # Хамгийн сүүлийн файлыг хадгалах (хэрэв олон байвал)
                                if plate_text not in self.plate_to_file:
                                    self.plate_to_file[plate_text] = os.path.abspath(file_path)
        except Exception as e:
            if self.debug_mode:
                print(f"⚠️  Файл ачаалах алдаа: {e}")

    def _detect_ocr_language(self):
        """Tesseract-д Монгол эсвэл Орос хэл байгаа эсэхийг шалгах"""
        try:
            available_langs = pytesseract.get_languages()
            if 'mon' in available_langs:
                print("✅ Монгол хэл (mon) олдлоо!")
                return 'mon'
            elif 'rus' in available_langs:
                print("⚠️  Монгол хэл (mon) олдсонгүй, Орос хэл (rus) ашиглаж байна.")
                print("   💡 Орос хэл нь Монгол кирилл үсгийг танина!")
                return 'rus'
            else:
                print("❌ Монгол (mon) болон Орос (rus) хэл олдсонгүй!")
                print("   💡 Tesseract-д 'rus.traineddata' эсвэл 'mon.traineddata' суулгана уу.")
                print("   📥 Татаж авах: https://github.com/tesseract-ocr/tessdata")
                return None
        except Exception as e:
            print(f"⚠️  Tesseract хэл шалгах алдаа: {e}")
            return None

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

    def format_video_time(self, seconds):
        """Секунд → MM:SS формат"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    def is_valid_plate(self, text):
        """Монгол дугаарын формат шалгах: Эхний 4 тоо + Сүүлийн 3 үсэг = 7 тэмдэгт"""
        if not text:
            return False
        
        # Яг 7 тэмдэгт байх ёстой
        if len(text) != 7:
            if self.debug_mode and self.debug_count < 10:
                print(f"   ❌ Урт буруу: {len(text)} (7 байх ёстой) - '{text}'")
            return False

        # Зөвхөн кирилл үсэг ба цифр байх ёстой
        if not text.isalnum():
            if self.debug_mode and self.debug_count < 10:
                print(f"   ❌ Тэмдэгт буруу: '{text}'")
            return False

        # Эхний 4 нь тоо байх ёстой
        first_four = text[:4]
        if not first_four.isdigit():
            if self.debug_mode and self.debug_count < 10:
                print(f"   ❌ Эхний 4 тоо биш: '{first_four}' in '{text}'")
            return False

        # Сүүлийн 3 нь үсэг байх ёстой
        last_three = text[4:]
        if not last_three.isalpha():
            if self.debug_mode and self.debug_count < 10:
                print(f"   ❌ Сүүлийн 3 үсэг биш: '{last_three}' in '{text}'")
            return False

        # Монгол кирилл үсэг эсэхийг шалгах
        mongolian_letters = set('АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯӨҮ')
        for char in last_three:
            if char not in mongolian_letters:
                if self.debug_mode and self.debug_count < 10:
                    print(f"   ❌ Монгол үсэг биш: '{char}' in '{last_three}'")
                return False

        return True

    def get_most_detected_plate(self):
        """Хамгийн олон удаа танигдсан дугаарыг олох"""
        if not self.detected_plates:
            return None, 0
        
        plate_counts = {}
        for det in self.detected_plates:
            plate = det['plate']
            plate_counts[plate] = plate_counts.get(plate, 0) + 1
        
        if not plate_counts:
            return None, 0
        
        most_common = max(plate_counts.items(), key=lambda x: x[1])
        return most_common[0], most_common[1]

    def is_duplicate(self, text, frame_number, video_time):
        """ИЛҮҮ сайн давхцал шалгах"""
        if text not in self.seen_plates:
            self.seen_plates[text] = {
                'frame': frame_number,
                'time': video_time,
                'count': 1
            }
            return False

        last_seen = self.seen_plates[text]
        frame_gap = frame_number - last_seen['frame']
        time_gap = video_time - last_seen['time']

        if frame_gap >= self.MIN_SAME_FRAME_GAP and time_gap >= self.MIN_SAME_TIME_GAP:
            self.seen_plates[text]['frame'] = frame_number
            self.seen_plates[text]['time'] = video_time
            self.seen_plates[text]['count'] += 1
            return False

        return True

    def detect_plates(self, frame):
        """ХУРДАН илрүүлэх"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        plates = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,  # Илүү нарийвчилсан (1.15-аас 1.1)
            minNeighbors=2,   # Багасгасан (3-аас 2)
            minSize=(50, 20)  # Багасгасан (70x25-аас 50x20)
        )

        valid = []
        for (x, y, w, h) in plates:
            ratio = w / h
            if 1.8 <= ratio <= 6.0:  # Илүү уян хатан (2.0-5.5-аас 1.8-6.0)
                valid.append((x, y, w, h))

        return valid

    def enhance_plate_fast(self, plate_img):
        """МААШГҮЙ ХУРДАН боловсруулалт"""
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()

        h, w = gray.shape

        target_h = 120
        scale = target_h / h
        gray = cv2.resize(gray, (int(w * scale), target_h),
                          interpolation=cv2.INTER_CUBIC)

        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def ocr_improved(self, img):
        """OCR - Монгол кирилл үсэг ба цифр танина (mon эсвэл rus хэл)."""
        if self.ocr_lang is None:
            if self.debug_mode and self.debug_count < 5:
                print("❌ OCR хэл тохируулагдаагүй байна!")
                self.debug_count += 1
            return None, 0
        
        # Зөвхөн Монгол кирилл том үсгүүд болон цифр
        mongolian_letters = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯӨҮ'
        whitelist = mongolian_letters + '0123456789'
        
        # Олон PSM режим туршиж үзэх
        psm_modes = [7, 8, 6, 11]  # 7=single line, 8=word, 6=block, 11=sparse text
        text = None
        best_text = None
        best_conf = 0
        
        for psm in psm_modes:
            config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist={whitelist}'
            try:
                text = pytesseract.image_to_string(img, config=config, lang=self.ocr_lang)
                if text and text.strip():
                    # Try to get confidence
                    try:
                        data = pytesseract.image_to_data(img, config=config, lang=self.ocr_lang, output_type=pytesseract.Output.DICT)
                        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                        avg_conf = sum(confidences) / len(confidences) if confidences else 0
                    except:
                        avg_conf = 50
                    
                    if avg_conf > best_conf:
                        best_text = text
                        best_conf = avg_conf
            except Exception as e:
                if self.ocr_lang == 'mon':
                    # mon байхгүй бол rus-д шилжих гэж оролдох
                    try:
                        text = pytesseract.image_to_string(img, config=config, lang='rus')
                        if text and text.strip():
                            self.ocr_lang = 'rus'
                            if self.debug_mode and self.debug_count < 3:
                                print("⚠️  Монгол хэл амжилтгүй, Орос хэл рүү шилжлээ.")
                                self.debug_count += 1
                            try:
                                data = pytesseract.image_to_data(img, config=config, lang='rus', output_type=pytesseract.Output.DICT)
                                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                                avg_conf = sum(confidences) / len(confidences) if confidences else 0
                            except:
                                avg_conf = 50
                            if avg_conf > best_conf:
                                best_text = text
                                best_conf = avg_conf
                    except:
                        continue
                else:
                    continue
        
        if not best_text:
            # Хэрэв whitelist-тэй амжилтгүй бол whitelist-гүйгээр туршиж үзэх
            try:
                config = f'--oem 3 --psm 7'
                text = pytesseract.image_to_string(img, config=config, lang=self.ocr_lang)
                if text and text.strip():
                    best_text = text
                    best_conf = 40
            except:
                pass
        
        if not best_text:
            if self.debug_mode and self.debug_count < 10:
                print(f"⚠️  OCR ямар ч текст олдсонгүй (хэл: {self.ocr_lang})")
                self.debug_count += 1
            return None, 0

        cleaned = self.clean_and_fix_text(best_text)
        
        if self.debug_mode and self.debug_count < 20:
            print(f"🔍 OCR raw: '{best_text}' -> cleaned: '{cleaned}' (conf: {best_conf:.1f})")
            if cleaned and not self.is_valid_plate(cleaned):
                print(f"   ⚠️  Validation failed: len={len(cleaned) if cleaned else 0}")
            self.debug_count += 1

        if cleaned and self.is_valid_plate(cleaned):
            digit_count = sum(c.isdigit() for c in cleaned)
            letter_count = sum(c.isalpha() for c in cleaned)

            if digit_count == 0 or letter_count == 0:
                balance = 0.0
            else:
                balance = min(digit_count, letter_count) / \
                    max(digit_count, letter_count)

            conf = max(40, min(90, best_conf * 0.8 + (balance * 20)))
            return cleaned, conf
        else:
            if self.debug_mode and self.debug_count < 10:
                if cleaned:
                    print(f"❌ Валидаци хийхэд амжилтгүй: '{cleaned}'")
                else:
                    print(f"❌ Текст цэвэрлэхэд амжилтгүй: '{best_text}'")
                self.debug_count += 1

        return None, 0

    def clean_and_fix_text(self, text):
        """Текст цэвэрлэх + Монгол дугаарын формат руу засах (4 тоо + 3 үсэг)."""
        if not text:
            return None

        text = text.strip().upper()

        # Зөвхөн Монгол кирилл том үсэг болон цифр үлдээх
        mongolian_letters = set('АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯӨҮ')
        allowed = mongolian_letters.union(set('0123456789'))

        cleaned = ''.join(c for c in text if c in allowed)

        if len(cleaned) < 6:  # Хамгийн багадаа 6 тэмдэгт (4 тоо + 2 үсэг)
            return None

        # Кирилл→цифр засварууд (OCR алдаа - эхний хэсэгт)
        corrections_to_digit = {
            'О': '0',
            'С': '5',
            'З': '3',
            'Б': '6',
            'И': '1',
            'Л': '1',
        }
        
        # Цифр→кирилл засварууд (OCR алдаа - сүүлийн хэсэгт)
        corrections_to_letter = {
            '0': 'О',
            '5': 'С',
            '3': 'З',
            '6': 'Б',
            '1': 'И',
        }

        # Тоонууд болон үсгүүдийг тусдаа хуваах
        digits = []
        letters = []
        ambiguous = []  # Засварлах шаардлагатай тэмдэгтүүд
        
        for c in cleaned:
            if c.isdigit():
                digits.append(c)
            elif c in mongolian_letters:
                letters.append(c)
            elif c in corrections_to_digit:
                # Засварлах шаардлагатай
                ambiguous.append((c, 'digit'))
            elif c in corrections_to_letter.values():
                # Засварлах шаардлагатай
                ambiguous.append((c, 'letter'))

        # Засварууд хийх - эхлээд тоонуудыг дүүргэх
        for char, target_type in ambiguous:
            if target_type == 'digit' and len(digits) < 4:
                digits.append(corrections_to_digit[char])
            elif target_type == 'letter' and len(letters) < 3:
                if char in corrections_to_letter:
                    letters.append(char)
                else:
                    # Урвуу засвар
                    for k, v in corrections_to_letter.items():
                        if v == char:
                            letters.append(char)
                            break

        # Хэрэв тоо хангалттай биш бол үсгүүдийг тоо болгох оролдлого
        if len(digits) < 4 and len(letters) > 3:
            for c in letters[:len(letters)-3]:
                if c in corrections_to_digit:
                    digits.append(corrections_to_digit[c])
                    letters.remove(c)
                    if len(digits) >= 4:
                        break

        # Хэрэв үсэг хангалттай биш бол тоонуудыг үсэг болгох оролдлого
        if len(letters) < 3 and len(digits) > 4:
            for c in digits[4:]:
                if c in corrections_to_letter:
                    letters.append(corrections_to_letter[c])
                    digits.remove(c)
                    if len(letters) >= 3:
                        break

        # Эхний 4 тоо
        first_four = ''.join(digits[:4])
        if len(first_four) < 4:
            return None

        # Сүүлийн 3 үсэг
        last_three = ''.join(letters[:3])
        if len(last_three) < 3:
            return None

        # Нийт 7 тэмдэгт
        result = first_four + last_three
        
        if len(result) != 7:
            return None

        return result

    def draw_plate_preview(self, frame):
        """Хамгийн олон удаа танигдсан дугаарын зургийг машин дээр харуулах"""
        # Хамгийн олон удаа танигдсан дугаарыг preview-д харуулах
        if self.most_detected_plate and self.most_detected_count > 1:
            # Хамгийн олон удаа танигдсан дугаарын зургийг олох
            if self.most_detected_plate in self.plate_to_file:
                file_path = self.plate_to_file[self.most_detected_plate]
                if os.path.exists(file_path):
                    try:
                        img = cv2.imread(file_path)
                        if img is not None:
                            self.last_plate_image = img
                            self.last_plate_text = self.most_detected_plate
                            # Хамгийн олон удаа танигдсан дугаарын дундаж confidence
                            confidences = [d['confidence'] for d in self.detected_plates 
                                         if d['plate'] == self.most_detected_plate]
                            if confidences:
                                self.last_plate_conf = sum(confidences) / len(confidences)
                            else:
                                self.last_plate_conf = 75
                    except:
                        pass
        
        if self.last_plate_image is None:
            return frame

        h, w = frame.shape[:2]

        # Preview хэсгийн байрлал (баруун доод буланд)
        preview_h = 150
        preview_w = 400
        margin = 20

        # Preview дэвсгэр байрлал
        preview_x = margin
        preview_y = h - preview_h - margin

        # Хар дэвсгэр зурах
        cv2.rectangle(frame,
                      (preview_x - 5, preview_y - 35),
                      (preview_x + preview_w + 5, preview_y + preview_h + 5),
                      (0, 0, 0), -1)

        # Ногоон хүрээ
        color = (0, 255, 0) if self.last_plate_conf >= 75 else (0, 220, 220)
        cv2.rectangle(frame,
                      (preview_x - 5, preview_y - 35),
                      (preview_x + preview_w + 5, preview_y + preview_h + 5),
                      color, 2)

        # Гарчиг
        if self.most_detected_plate and self.last_plate_text == self.most_detected_plate:
            title = f"HAMGIIN OLON ({self.most_detected_count}x):"
            title_color = (0, 255, 255)  # Шар
        else:
            title = "SUULIIN DUGAAR:"
            title_color = (255, 255, 255)  # Цагаан
        
        cv2.putText(frame, title,
                    (preview_x + 5, preview_y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, title_color, 2)

        # Дугаарын зургийг багасгаж preview хэсэгт зурах
        plate_img = self.last_plate_image.copy()

        # Зургийн хэмжээг тооцоолох
        img_h, img_w = plate_img.shape[:2]
        scale = min(preview_w / img_w, preview_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        # Resize хийх
        resized_plate = cv2.resize(plate_img, (new_w, new_h))

        # Төвлөрүүлэх
        offset_x = preview_x + (preview_w - new_w) // 2
        offset_y = preview_y + (preview_h - new_h) // 2

        # Зургийг байршуулах
        frame[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized_plate

        # Дугаар болон confidence-ийг доор харуулах
        if self.last_plate_text:
            text_y = preview_y + preview_h + 25

            # Дэвсгэр
            label = f"{self.last_plate_text} ({self.last_plate_conf:.0f}%)"
            # Текстийн хэмжээг тооцоолох (PIL ашиглах)
            if PIL_AVAILABLE:
                try:
                    from PIL import ImageFont
                    font_size = int(0.9 * 40)
                    font_paths = [
                        r"C:\Windows\Fonts\arial.ttf",
                        r"C:\Windows\Fonts\calibri.ttf",
                        r"C:\Windows\Fonts\tahoma.ttf",
                    ]
                    font = None
                    for font_path in font_paths:
                        if os.path.exists(font_path):
                            font = ImageFont.truetype(font_path, font_size)
                            break
                    if font:
                        bbox = font.getbbox(label)
                        txt_w = bbox[2] - bbox[0]
                        txt_h = bbox[3] - bbox[1]
                    else:
                        (txt_w, txt_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
                except:
                    (txt_w, txt_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
            else:
                (txt_w, txt_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)

            text_x = preview_x + (preview_w - txt_w) // 2

            cv2.rectangle(frame,
                          (text_x - 10, text_y - txt_h - 8),
                          (text_x + txt_w + 10, text_y + 5),
                          (0, 0, 0), -1)

            # Кирилл текстийг зөв харуулах
            self.put_text_cyrillic(frame, label, (text_x, text_y), font_scale=0.9, color=color, thickness=2)

        return frame

    def draw_table(self, frame, video_fps):
        """Хүснэгт - VIDEO цаг харуулах + Дарах боломжтой бүсүүд"""
        h, w = frame.shape[:2]
        table_w = 380
        table_x = w - table_w

        cv2.rectangle(frame, (table_x, 0), (w, h), (18, 18, 18), -1)

        cv2.rectangle(frame, (table_x, 0), (w, 55), (0, 100, 0), -1)
        cv2.putText(frame, "TANISAN DUGAARUD",
                    (table_x + 60, 37), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

        y = 75
        cv2.line(frame, (table_x + 10, y), (w - 10, y), (80, 80, 80), 2)
        y += 28

        cv2.putText(frame, "#", (table_x + 15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(frame, "Video Tsag", (table_x + 50, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(frame, "Dugaar", (table_x + 160, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(frame, "%", (table_x + 320, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        y += 8
        cv2.line(frame, (table_x + 10, y), (w - 10, y), (60, 60, 60), 1)

        # Clickable regions-ийг цэвэрлэх
        self.clickable_regions = {}
        
        # Хамгийн олон удаа танигдсан дугаарыг олох
        most_plate, most_count = self.get_most_detected_plate()
        self.most_detected_plate = most_plate
        self.most_detected_count = most_count
        
        start_idx = max(0, len(self.detected_plates) - 9)
        for i, det in enumerate(self.detected_plates[start_idx:], start=start_idx+1):
            y += 38
            if y > h - 80:
                break

            time_str = self.format_video_time(det['video_time'])
            plate = det['plate']
            conf = det['confidence']

            # Хамгийн олон удаа танигдсан дугаарыг онцолж харуулах
            if plate == most_plate and most_count > 1:
                color = (0, 255, 255)  # Шар (онцолсон)
                # Онцолсон хүрээ зурах
                cv2.rectangle(frame, (table_x + 5, y - 25), 
                            (w - 5, y + 10), (0, 255, 255), 2)
                # "HAMGIIN OLON" текст
                cv2.putText(frame, "HAMGIIN OLON", (table_x + 10, y - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            else:
                color = (0, 255, 0) if conf >= 75 else (0, 220, 220)

            cv2.putText(frame, f"{i}", (table_x + 15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            cv2.putText(frame, time_str, (table_x + 50, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Дугаарын бүс - дарах боломжтой
            plate_x = table_x + 160
            plate_y = y - 20
            plate_w = 200
            plate_h = 30
            
            # Дарах боломжтой бүс (дугаарын текст дээр)
            if plate in self.plate_to_file:
                # Хөх хүрээ зурах (дарах боломжтойг илтгэх)
                cv2.rectangle(frame, (plate_x - 5, plate_y), 
                            (plate_x + plate_w, plate_y + plate_h), (100, 150, 255), 1)
                # Clickable region хадгалах
                self.clickable_regions[(plate_x - 5, plate_y, plate_x + plate_w, plate_y + plate_h)] = plate

            # Кирилл текстийг зөв харуулах
            self.put_text_cyrillic(frame, plate, (plate_x, y), font_scale=0.7, color=color, thickness=2)

            cv2.putText(frame, f"{conf:.0f}", (table_x + 320, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        cv2.rectangle(frame, (table_x, h-60), (w, h), (28, 28, 28), -1)
        cv2.putText(frame, f"Niit olson: {len(self.detected_plates)}",
                    (table_x + 20, h - 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

        unique = len(set(d['plate'] for d in self.detected_plates))
        cv2.putText(frame, f"Unique: {unique}",
                    (table_x + 20, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)
        
        # Хамгийн олон удаа танигдсан дугаарын мэдээлэл
        if self.most_detected_plate and self.most_detected_count > 1:
            info_y = h - 200
            cv2.putText(frame, f"HAMGIIN OLON: {self.most_detected_plate}",
                        (table_x + 20, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"({self.most_detected_count} udaa)",
                        (table_x + 20, info_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        
        # Заавар текст
        if len(self.clickable_regions) > 0:
            cv2.putText(frame, "Dugaar deer darah -> file neeh", (table_x + 20, h - 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)

        return frame

    def draw_detection(self, frame, x, y, w, h, text, conf):
        """Илрүүлэлт зурах"""
        color = (0, 255, 0) if conf >= 75 else (0, 200, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        if text:
            label = f"{text} ({conf:.0f}%)"
            (txt_w, txt_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            cv2.rectangle(frame, (x, y-28), (x+txt_w+10, y), (0, 0, 0), -1)
            cv2.rectangle(frame, (x, y-28), (x+txt_w+10, y), color, 2)
            # Кирилл текстийг зөв харуулах
            self.put_text_cyrillic(frame, label, (x+5, y-8), font_scale=0.6, color=color, thickness=2)

        return frame

    def save_result(self, plate_img, text, video_time, is_low=False):
        """Хадгалах - алдаа шалгахтай
        
        Args:
            plate_img: Дугаарын зураг
            text: Дугаарын текст
            video_time: Видеоны цаг
            is_low: True бол _LOW_ prefix нэмэх (хамгийн олон биш дугаар)
        """
        try:
            # Хавтас байгаа эсэхийг шалгах
            if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder)
            
            time_str = self.format_video_time(video_time)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Файлын нэрэнд хориглосон тэмдэгтүүдийг засах
            safe_text = "".join(c for c in text if c.isalnum() or c in ('-', '_'))
            
            # _LOW_ prefix нэмэх эсвэл үгүй
            if is_low:
                filename = f"_LOW_{safe_text}_{time_str.replace(':', '-')}_{timestamp}.jpg"
            else:
                filename = f"{safe_text}_{time_str.replace(':', '-')}_{timestamp}.jpg"
            
            img_file = os.path.join(self.save_folder, filename)
            
            # Зургийг хадгалах
            success = cv2.imwrite(img_file, plate_img)
            
            if success:
                # Зөвхөн хамгийн олон дугаарын файлын замыг хадгалах
                if not is_low:
                    self.plate_to_file[text] = os.path.abspath(img_file)
                    if self.debug_mode:
                        print(f"💾 Хадгалсан (ХАМГИЙН ОЛОН): {filename}")
                else:
                    if self.debug_mode:
                        print(f"💾 Хадгалсан (_LOW_): {filename}")
            else:
                print(f"❌ Хадгалах амжилтгүй: {filename}")
                
        except Exception as e:
            print(f"❌ Хадгалах алдаа: {e}")
    
    def open_file(self, file_path):
        """Файлыг системийн default програм дээр нээх"""
        try:
            if platform.system() == 'Windows':
                os.startfile(file_path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.call(['open', file_path])
            else:  # Linux
                subprocess.call(['xdg-open', file_path])
        except Exception as e:
            print(f"❌ Файл нээх алдаа: {e}")


def main():
    print("\n" + "="*70)
    print(" "*10 + "🚗 ВИДЕО ДУГААР ТАНИХ (МОНГОЛ ҮСЭГ) 🚗")
    print("="*70)
    print("\n✨ Сайжруулалт:")
    print("  • ⭐ МОНГОЛ КИРИЛЛ ҮСЭГ ТАНИЛТ (mon эсвэл rus хэл)")
    print("  • Хурдан ажиллана (10 frame skip)")
    print("  • Давхцал сайн шалгана (60 frame gap)")
    print("  • Буруу танилт засна (O→0, I→1, гэх мэт)")
    print("  • Видеоны цаг харуулна (MM:SS)")
    print("  • Unique дугаар тоолно")
    print("  • Сүүлийн дугаарын зургийг машин дээр харуулна!")
    print("  • 🆕 _LOW_ файл систем - зөвхөн хамгийн олон дугаар хадгална!")
    print("\n" + "-"*70 + "\n")

    detector = FastPlateDetector()
    
    # Тохиргооны мэдээлэл
    print(f"\n📋 Тохиргоо:")
    print(f"   OCR хэл: {detector.ocr_lang if detector.ocr_lang else 'ОЛДСОНГҮЙ!'}")
    print(f"   MIN_CONFIDENCE: {detector.MIN_CONFIDENCE}%")
    print(f"   Debug mode: {'ON' if detector.debug_mode else 'OFF'}")
    if detector.ocr_lang is None:
        print("\n⚠️  АНХААР: OCR хэл олдсонгүй! Дугаар таних боломжгүй!")
        print("   Tesseract-д 'rus.traineddata' эсвэл 'mon.traineddata' суулгана уу.")
    print()

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

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps

    print(f"📊 Мэдээлэл:")
    print(f"   FPS: {fps:.1f}")
    print(f"   Resolution: {width}x{height}")
    print(f"   Frames: {total_frames}")
    print(f"   Үргэлжлэх: {detector.format_video_time(duration)}\n")

    display_w = min(width, 1280)
    display_h = int(height * (display_w / width))

    print("🚀 Эхэлж байна...\n")
    print("   SPACE - Зогсоох/Үргэлжлүүлэх")
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
            video_time = frame_count / fps

            if frame.shape[1] != display_w:
                frame = cv2.resize(frame, (display_w, display_h))

            if frame_count % 10 == 0:
                plates = detector.detect_plates(frame)
                
                if detector.debug_mode and len(plates) > 0 and detector.debug_count < 20:
                    print(f"🔍 Frame {frame_count}: {len(plates)} plate(s) detected")

                for (x, y, w, h) in plates:
                    plate_img = frame[y:y+h, x:x+w]
                    
                    if plate_img.size == 0:
                        continue

                    enhanced = detector.enhance_plate_fast(plate_img)
                    text, conf = detector.ocr_improved(enhanced)

                    if text:
                        # Бүх танигдсан дугааруудыг detected_plates-д нэмэх (confidence-ээс үл хамааран)
                        is_new_detection = not detector.is_duplicate(text, frame_count, video_time)
                        
                        if is_new_detection:
                            # Сүүлийн дугаарыг хадгалах (preview-д харуулах)
                            detector.last_plate_image = plate_img.copy()
                            detector.last_plate_text = text
                            detector.last_plate_conf = conf

                            detector.detected_plates.append({
                                'plate': text,
                                'confidence': conf,
                                'video_time': video_time,
                                'frame': frame_count
                            })
                            
                            # Хамгийн олон удаа танигдсан дугаарыг шинэчлэх
                            most_plate, most_count = detector.get_most_detected_plate()
                            old_most_plate = detector.most_detected_plate
                            old_most_count = detector.most_detected_count
                            
                            # _LOW_ файлууд хадгалах (хамгийн олон биш дугаарууд)
                            if text != most_plate or most_count <= 1:
                                detector.save_result(plate_img, text, video_time, is_low=True)
                            
                            # Хамгийн олон удаа танигдсан дугаарыг шинэчлэх
                            if most_count > detector.most_detected_count:
                                detector.most_detected_plate = most_plate
                                detector.most_detected_count = most_count
                                print(f"⭐ ХАМГИЙН ОЛОН: {most_plate} ({most_count} удаа)")
                                
                                # Зөвхөн хамгийн олон удаа танигдсан дугаарыг хадгалах
                                if old_most_plate != most_plate:
                                    # Хуучин хамгийн олон дугаарын файлыг _LOW_ руу шилжүүлэх
                                    if old_most_plate and old_most_plate in detector.plate_to_file:
                                        old_file = detector.plate_to_file[old_most_plate]
                                        try:
                                            if os.path.exists(old_file):
                                                # _LOW_ prefix нэмэх
                                                dirname = os.path.dirname(old_file)
                                                basename = os.path.basename(old_file)
                                                new_name = f"_LOW_{basename}"
                                                new_path = os.path.join(dirname, new_name)
                                                os.rename(old_file, new_path)
                                                print(f"📦 Хуучин дугаарыг _LOW_ руу шилжүүлсэн: {basename}")
                                                # plate_to_file-оос устгах
                                                del detector.plate_to_file[old_most_plate]
                                        except Exception as e:
                                            print(f"⚠️  Файл шилжүүлэх алдаа: {e}")
                                    
                                    # Шинэ хамгийн олон дугаарыг хадгалах
                                    if text == most_plate:
                                        detector.save_result(plate_img, text, video_time, is_low=False)
                            elif most_count == detector.most_detected_count and most_plate == detector.most_detected_plate:
                                # Хамгийн олон удаа танигдсан дугаар ижил хэвээр байна
                                # Файл байхгүй эсвэл шинэчлэх шаардлагатай бол хадгалах
                                if most_plate not in detector.plate_to_file or not os.path.exists(detector.plate_to_file.get(most_plate, '')):
                                    if text == most_plate:
                                        detector.save_result(plate_img, text, video_time, is_low=False)
                            elif old_most_count == 0 and most_count >= 1:
                                # Эхний удаа хамгийн олон удаа танигдсан дугаар тодорхойлогдож байна
                                detector.most_detected_plate = most_plate
                                detector.most_detected_count = most_count
                                if text == most_plate:
                                    detector.save_result(plate_img, text, video_time, is_low=False)
                                print(f"⭐ ХАМГИЙН ОЛОН: {most_plate} ({most_count} удаа)")

                            time_str = detector.format_video_time(video_time)
                            print(
                                f"✅ {len(detector.detected_plates)}. {text} ({conf:.0f}%) @ {time_str}")

                        # Зургийг харуулах (confidence-ээс үл хамааран)
                        if conf >= detector.MIN_CONFIDENCE:
                            frame = detector.draw_detection(
                                frame, x, y, w, h, text, conf)
                        else:
                            # Confidence бага - зөвхөн харагдах, хадгалахгүй
                            frame = detector.draw_detection(
                                frame, x, y, w, h, f"{text} ({conf:.0f}%)", conf)
                    else:
                        # Plate илрүүлсэн боловч OCR амжилтгүй
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, "OCR failed", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Хүснэгт
            frame = detector.draw_table(frame, fps)

            # Сүүлийн дугаарын preview харуулах
            frame = detector.draw_plate_preview(frame)

            # Статус
            h_frame = frame.shape[0]
            cv2.rectangle(frame, (5, h_frame-50),
                          (350, h_frame-5), (25, 25, 25), -1)

            curr_time_str = detector.format_video_time(video_time)
            total_time_str = detector.format_video_time(duration)
            cv2.putText(frame, f"Video: {curr_time_str} / {total_time_str}",
                        (10, h_frame-28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)

            progress = (frame_count / total_frames) * 100
            cv2.putText(frame, f"Progress: {progress:.1f}%",
                        (10, h_frame-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

        cv2.imshow('Video Plate Detection', frame)
        
        # Mouse callback-ийг тохируулах (цонх үүссэний дараа, зөвхөн нэг удаа)
        if not hasattr(detector, 'mouse_callback_set'):
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    # Clickable region-үүдийг шалгах
                    for (x1, y1, x2, y2), plate in detector.clickable_regions.items():
                        if x1 <= x <= x2 and y1 <= y <= y2:
                            if plate in detector.plate_to_file:
                                file_path = detector.plate_to_file[plate]
                                if os.path.exists(file_path):
                                    print(f"🖼️  Файл нээж байна: {os.path.basename(file_path)}")
                                    detector.open_file(file_path)
                                else:
                                    print(f"❌ Файл олдсонгүй: {file_path}")
                            break
            
            try:
                cv2.setMouseCallback('Video Plate Detection', mouse_callback)
                detector.mouse_callback_set = True
                print("🖱️  Mouse callback тохируулсан - Дугаар дээр дараад файл нээх боломжтой!")
            except cv2.error as e:
                # Цонх хараахгүй байвал дараагийн frame дээр дахин оролдох
                pass

        key = cv2.waitKey(1) & 0xFF
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
    print(f"Нийт танигдсан: {len(detector.detected_plates)}")

    if detector.detected_plates:
        unique_plates = set(d['plate'] for d in detector.detected_plates)
        print(f"Unique дугаар: {len(unique_plates)}")

        print(f"\n📋 Бүх дугаарууд:")
        for i, det in enumerate(detector.detected_plates, 1):
            time_str = detector.format_video_time(det['video_time'])
            print(
                f"  {i}. {det['plate']} ({det['confidence']:.0f}%) @ {time_str}")

        avg_conf = sum(d['confidence']
                       for d in detector.detected_plates) / len(detector.detected_plates)
        print(f"\nДундаж confidence: {avg_conf:.1f}%")

    # Хадгалагдсан файлуудын тоо
    save_folder_path = os.path.abspath(detector.save_folder)
    if os.path.exists(save_folder_path):
        all_files = [f for f in os.listdir(save_folder_path) if f.endswith('.jpg')]
        main_files = [f for f in all_files if not f.startswith('_LOW_')]
        low_files = [f for f in all_files if f.startswith('_LOW_')]
        
        print(f"\n💾 Хадгалагдсан файлууд:")
        print(f"   Хамгийн олон дугаарууд: {len(main_files)}")
        print(f"   _LOW_ файлууд: {len(low_files)}")
        print(f"   Нийт: {len(all_files)}")
        print(f"   Хавтас: {save_folder_path}")
        
        if main_files:
            print(f"\n   ⭐ Хамгийн олон дугаарууд:")
            for f in main_files[:5]:
                print(f"     - {f}")
            if len(main_files) > 5:
                print(f"     ... ба {len(main_files) - 5} файл")
    else:
        print(f"\n⚠️  Хавтас олдсонгүй: {save_folder_path}")
    
    print("\n👋 Баяртай!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()