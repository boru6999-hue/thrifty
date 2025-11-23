import os
import cv2  # ← ЭНЭ МӨРИЙГ НЭМЭХ!
from pathlib import Path


class Config:
    """Configuration class for plate detector"""

    # OCR параметрүүд
    MIN_CONFIDENCE = 50
    MIN_SAME_FRAME_GAP = 60
    MIN_SAME_TIME_GAP = 3.0
    DEBUG_MODE = True

    # Tesseract зам
    TESSERACT_PATHS = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    ]

    # Хавтсууд
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
    DETECTED_PLATES_DIR = os.path.join(OUTPUT_DIR, 'detected_plates')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')

    # Cascade classifier - OpenCV-ийн built-in ашиглах
    CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'

    # Шрифтын замууд
    FONT_PATHS = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\tahoma.ttf",
    ]

    # Plate параметрүүд
    PLATE_MIN_RATIO = 1.8
    PLATE_MAX_RATIO = 6.0
    PLATE_MIN_SIZE = (50, 20)
    PLATE_TARGET_HEIGHT = 120

    # Video параметрүүд
    FRAME_SKIP = 10
    DISPLAY_MAX_WIDTH = 1280

    @staticmethod
    def ensure_directories():
        """Шаардлагатай хавтсууд үүсгэх"""
        for directory in [Config.OUTPUT_DIR, Config.DETECTED_PLATES_DIR, Config.LOGS_DIR]:
            os.makedirs(directory, exist_ok=True)


# Initialize directories on module load
Config.ensure_directories()
