# src/utils.py
import cv2
import numpy as np
import os

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from .config import Config


def format_video_time(seconds):
    """Секунд → MM:SS формат"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def is_valid_plate(text):
    """Монгол дугаарын формат шалгах: 4 тоо + 3 үсэг = 7 тэмдэгт"""
    if not text or len(text) != 7:
        return False

    if not text.isalnum():
        return False

    # Эхний 4 нь тоо
    if not text[:4].isdigit():
        return False

    # Сүүлийн 3 нь үсэг
    if not text[4:].isalpha():
        return False

    # Монгол кирилл үсэг
    mongolian_letters = set('АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯӨҮ')
    for char in text[4:]:
        if char not in mongolian_letters:
            return False

    return True


def put_text_cyrillic(frame, text, position, font_scale=0.7, color=(255, 255, 255), thickness=2):
    """Кирилл текстийг зөв харуулах (PIL ашиглах)"""
    if not PIL_AVAILABLE:
        try:
            cv2.putText(frame, text.encode('ascii', 'replace').decode('ascii'),
                        position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        except:
            cv2.putText(frame, "???", position,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        return

    try:
        x, y = position
        font_size = int(font_scale * 40)

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        font = None
        for font_path in Config.FONT_PATHS:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break

        if font is None:
            font = ImageFont.load_default()

        rgb_color = (color[2], color[1], color[0])
        draw.text((x, y), text, font=font, fill=rgb_color)

        frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        try:
            cv2.putText(frame, text.encode('ascii', 'replace').decode('ascii'),
                        position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        except:
            cv2.putText(frame, "???", position,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
