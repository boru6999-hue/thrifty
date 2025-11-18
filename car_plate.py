import cv2
import numpy as np
import pytesseract
from datetime import datetime, timedelta
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
        self.seen_plates = {}

        # Сүүлд илэрсэн дугаарын зураг хадгалах
        self.last_plate_image = None
        self.last_plate_text = None
        self.last_plate_conf = None

        # Шалгуур
        self.MIN_CONFIDENCE = 65
        self.MIN_SAME_FRAME_GAP = 60
        self.MIN_SAME_TIME_GAP = 3.0

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

    def format_video_time(self, seconds):
        """Секунд → MM:SS формат"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    def is_valid_plate(self, text):
        """Дугаарын формат - ХАТУУ шалгах"""
        if not text or len(text) < 5 or len(text) > 10:
            return False

        if not text.isalnum():
            return False

        digit_count = sum(c.isdigit() for c in text)
        if digit_count < 2:
            return False

        letter_count = sum(c.isalpha() for c in text)
        if letter_count < 1:
            return False

        if digit_count == 0 or letter_count == 0:
            return False

        return True

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
            scaleFactor=1.15,
            minNeighbors=3,
            minSize=(70, 25)
        )

        valid = []
        for (x, y, w, h) in plates:
            ratio = w / h
            if 2.0 <= ratio <= 5.5:
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
        """OCR - сайжруулсан буруу танилтыг засах"""
        try:
            config = '--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789АБВГДЕЖЗИЙКЛМНӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯ'

            text = pytesseract.image_to_string(img, config=config, lang='eng')
            cleaned = self.clean_and_fix_text(text)

            if cleaned and self.is_valid_plate(cleaned):
                digit_count = sum(c.isdigit() for c in cleaned)
                letter_count = sum(c.isalpha() for c in cleaned)

                balance = min(digit_count, letter_count) / \
                    max(digit_count, letter_count)
                conf = 60 + (balance * 30)

                return cleaned, conf
        except Exception as e:
            pass

        return None, 0

    def clean_and_fix_text(self, text):
        """Текст цэвэрлэх + БУРУУ тэмдэгт засах"""
        text = text.strip()
        cleaned = ''.join(c for c in text if c.isalnum())
        cleaned = cleaned.upper()

        corrections = {
            'O': '0',
            'I': '1',
            'S': '5',
            'Z': '2',
            'B': '8',
            'G': '6',
        }

        digit_count = sum(c.isdigit() for c in cleaned)
        total = len(cleaned)

        if total > 0 and digit_count / total > 0.5:
            result = []
            for c in cleaned:
                if c in corrections and c.isalpha():
                    result.append(corrections[c])
                else:
                    result.append(c)
            cleaned = ''.join(result)

        if len(cleaned) < 5 or len(cleaned) > 10:
            return None

        return cleaned

    def draw_plate_preview(self, frame):
        """Сүүлд илэрсэн дугаарын зургийг машин дээр харуулах"""
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
        cv2.putText(frame, "SUULIIN DUGAAR:",
                    (preview_x + 5, preview_y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

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
            (txt_w, txt_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)

            text_x = preview_x + (preview_w - txt_w) // 2

            cv2.rectangle(frame,
                          (text_x - 10, text_y - txt_h - 8),
                          (text_x + txt_w + 10, text_y + 5),
                          (0, 0, 0), -1)

            cv2.putText(frame, label,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)

        return frame

    def draw_table(self, frame, video_fps):
        """Хүснэгт - VIDEO цаг харуулах"""
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

        start_idx = max(0, len(self.detected_plates) - 9)
        for i, det in enumerate(self.detected_plates[start_idx:], start=start_idx+1):
            y += 38
            if y > h - 80:
                break

            time_str = self.format_video_time(det['video_time'])
            plate = det['plate']
            conf = det['confidence']

            color = (0, 255, 0) if conf >= 75 else (0, 220, 220)

            cv2.putText(frame, f"{i}", (table_x + 15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            cv2.putText(frame, time_str, (table_x + 50, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.putText(frame, plate, (table_x + 160, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

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
            cv2.putText(frame, label, (x+5, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    def save_result(self, plate_img, text, video_time):
        """Хадгалах"""
        time_str = self.format_video_time(video_time)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"{text}_{time_str.replace(':', '-')}_{timestamp}.jpg"
        img_file = os.path.join(self.save_folder, filename)
        cv2.imwrite(img_file, plate_img)


def main():
    print("\n" + "="*70)
    print(" "*10 + "🚗 ВИДЕО ДУГААР ТАНИХ (SCREENSHOT ХАРУУЛАХ) 🚗")
    print("="*70)
    print("\n✨ Сайжруулалт:")
    print("  • Хурдан ажиллана (10 frame skip)")
    print("  • Давхцал сайн шалгана (60 frame gap)")
    print("  • Буруу танилт засна (O→0, I→1, гэх мэт)")
    print("  • Видеоны цаг харуулна (MM:SS)")
    print("  • Unique дугаар тоолно")
    print("  • ⭐ ШИНЭ: Сүүлийн дугаарын зургийг машин дээр харуулна!")
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

                for (x, y, w, h) in plates:
                    plate_img = frame[y:y+h, x:x+w]

                    enhanced = detector.enhance_plate_fast(plate_img)
                    text, conf = detector.ocr_improved(enhanced)

                    if text and conf >= detector.MIN_CONFIDENCE:
                        if not detector.is_duplicate(text, frame_count, video_time):
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

                            detector.save_result(plate_img, text, video_time)

                            time_str = detector.format_video_time(video_time)
                            print(
                                f"✅ {len(detector.detected_plates)}. {text} ({conf:.0f}%) @ {time_str}")

                        frame = detector.draw_detection(
                            frame, x, y, w, h, text, conf)

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

    print(f"\n💾 Файлууд: {detector.save_folder}/")
    print("\n👋 Баяртай!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
