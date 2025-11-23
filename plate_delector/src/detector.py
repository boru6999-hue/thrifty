# src/detector.py
import cv2
import numpy as np
import os
import shutil
import pytesseract
from collections import deque

from .config import Config
from .ocr import OCRHandler
from .utils import format_video_time, is_valid_plate, put_text_cyrillic
from .file_handler import FileHandler
# detector.py дотор _setup_tesseract() функцын дараа:


def __init__(self):
    # ... бусад код

    # Cascade classifier - OpenCV-ийн built-in ашиглах
    cascade_path = Config.CASCADE_PATH
    if not os.path.exists(cascade_path):
        # OpenCV-ийн өөрийн cascade ашиглах
        import cv2
        cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'

    self.cascade = cv2.CascadeClassifier(cascade_path)

    if self.cascade.empty():
        print("⚠️ Cascade файл ачаалагдсангүй!")


class FastPlateDetector:
    def __init__(self):
        Config.ensure_directories()

        self.file_handler = FileHandler()
        self.ocr_handler = OCRHandler()

        print("✅ Хурдан систем бэлэн!")

        self.file_handler = FileHandler()
        self.ocr_handler = OCRHandler()

        self.save_folder = Config.DETECTED_PLATES_DIR
        self.detected_plates = []
        self.seen_plates = {}

        self.last_plate_image = None
        self.last_plate_text = None
        self.last_plate_conf = None

        self.most_detected_plate = None
        self.most_detected_count = 0

        self.MIN_CONFIDENCE = Config.MIN_CONFIDENCE
        self.MIN_SAME_FRAME_GAP = Config.MIN_SAME_FRAME_GAP
        self.MIN_SAME_TIME_GAP = Config.MIN_SAME_TIME_GAP

        self.debug_mode = Config.DEBUG_MODE
        self.debug_count = 0

        self.clickable_regions = {}
        self.plate_to_file = self.file_handler.plate_to_file

        # Tesseract зам
        self._setup_tesseract()

        # Cascade classifier
        self.cascade = cv2.CascadeClassifier(Config.CASCADE_PATH)

        print("✅ Хурдан систем бэлэн!")

    def _setup_tesseract(self):
        """Tesseract зам тохируулах"""
        tess_default = Config.TESSERACT_PATHS[0]
        if os.path.exists(tess_default):
            pytesseract.pytesseract.tesseract_cmd = tess_default
        else:
            t_in_path = shutil.which('tesseract')
            if t_in_path:
                pytesseract.pytesseract.tesseract_cmd = t_in_path
            else:
                print("⚠️ Tesseract not found. Please install Tesseract OCR.")

    def detect_plates(self, frame):
        """Дугаар илрүүлэх"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        plates = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=Config.PLATE_MIN_SIZE
        )

        valid = []
        for (x, y, w, h) in plates:
            ratio = w / h
            if Config.PLATE_MIN_RATIO <= ratio <= Config.PLATE_MAX_RATIO:
                valid.append((x, y, w, h))

        return valid

    def enhance_plate_fast(self, plate_img):
        """Дугаарын зургийг сайжруулах"""
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img.copy()

        h, w = gray.shape

        target_h = Config.PLATE_TARGET_HEIGHT
        scale = target_h / h
        gray = cv2.resize(gray, (int(w * scale), target_h),
                          interpolation=cv2.INTER_CUBIC)

        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def ocr_plate(self, img):
        """OCR хийх"""
        return self.ocr_handler.ocr_improved(img)

    def is_duplicate(self, text, frame_number, video_time):
        """Давхцал шалгах"""
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
            put_text_cyrillic(frame, label, (x+5, y-8),
                              font_scale=0.6, color=color, thickness=2)

        return frame

    def draw_plate_preview(self, frame):
        """Сүүлийн дугаарын preview-г харуулах"""
        if self.most_detected_plate and self.most_detected_plate in self.plate_to_file:
            file_path = self.plate_to_file[self.most_detected_plate]
            if os.path.exists(file_path):
                try:
                    img = cv2.imread(file_path)
                    if img is not None:
                        self.last_plate_image = img
                except:
                    pass

        if self.last_plate_image is None:
            return frame

        h, w = frame.shape[:2]
        preview_h, preview_w = 150, 400
        margin = 20
        preview_x = margin
        preview_y = h - preview_h - margin

        cv2.rectangle(frame, (preview_x - 5, preview_y - 35),
                      (preview_x + preview_w + 5, preview_y + preview_h + 5), (0, 0, 0), -1)

        color = (0, 255, 0) if self.last_plate_conf and self.last_plate_conf >= 75 else (
            0, 220, 220)
        cv2.rectangle(frame, (preview_x - 5, preview_y - 35),
                      (preview_x + preview_w + 5, preview_y + preview_h + 5), color, 2)

        title = f"HAMGIIN OLON ({self.most_detected_count}x):" if self.most_detected_plate else "SUULIIN DUGAAR:"
        cv2.putText(frame, title, (preview_x + 5, preview_y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        plate_img = self.last_plate_image.copy()
        img_h, img_w = plate_img.shape[:2]
        scale = min(preview_w / img_w, preview_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)

        resized_plate = cv2.resize(plate_img, (new_w, new_h))
        offset_x = preview_x + (preview_w - new_w) // 2
        offset_y = preview_y + (preview_h - new_h) // 2

        frame[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized_plate

        if self.last_plate_text:
            text_y = preview_y + preview_h + 25
            label = f"{self.last_plate_text} ({self.last_plate_conf:.0f}%)" if self.last_plate_conf else self.last_plate_text
            (txt_w, txt_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
            text_x = preview_x + (preview_w - txt_w) // 2

            cv2.rectangle(frame, (text_x - 10, text_y - txt_h - 8),
                          (text_x + txt_w + 10, text_y + 5), (0, 0, 0), -1)

            put_text_cyrillic(frame, label, (text_x, text_y),
                              font_scale=0.9, color=color, thickness=2)

        return frame
