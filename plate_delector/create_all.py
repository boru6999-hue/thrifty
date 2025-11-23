# create_all.py
# Энэ файлыг VS Code-д plate-detector хавтасанд үүсгээнэ
# python create_all.py

import os

print("=" * 70)
print("🔧 БҮХЭЛ БҮ SETUP ЭХЭЛЖ БАЙНА")
print("=" * 70 + "\n")

# Folder үүсгэх
print("📁 Папкууд үүсгэж байна...")
os.makedirs('src', exist_ok=True)
os.makedirs('resources/cascades', exist_ok=True)
os.makedirs('output/detected_plates', exist_ok=True)
os.makedirs('logs', exist_ok=True)
print("✅ Папкууд үүсгэгдсэн\n")

# 1. requirements.txt
print("📝 requirements.txt үүсгэж байна...")
with open('requirements.txt', 'w') as f:
    f.write("""opencv-python==4.8.0.74
pytesseract==0.3.10
Pillow==10.0.0
numpy==1.24.0
""")
print("✅ requirements.txt үүсгэгдсэн\n")

# 2. src/__init__.py
print("📝 src/__init__.py үүсгэж байна...")
with open('src/__init__.py', 'w') as f:
    f.write("""# src/__init__.py
from .detector import FastPlateDetector
from .ocr import OCRHandler
from .file_handler import FileHandler
from .config import Config
from .utils import format_video_time, is_valid_plate, put_text_cyrillic

__all__ = [
    'FastPlateDetector',
    'OCRHandler', 
    'FileHandler',
    'Config',
    'format_video_time',
    'is_valid_plate',
    'put_text_cyrillic'
]
""")
print("✅ src/__init__.py үүсгэгдсэн\n")

print("=" * 70)
print("✅ ҮНДСЭН SETUP ДУУСАА!")
print("=" * 70)
print("\n📋 ДАРААГИЙН АЛХАМ:\n")
print("1️⃣  Доорх ФАЙЛУУДЫГ VS CODE-Д COPY ХИЙНЭ:")
print("   (src/ папка дотор)")
print("   • config.py")
print("   • ocr.py")
print("   • utils.py")
print("   • file_handler.py")
print("   • detector.py")
print("\n2️⃣  Доорх ФАЙЛУУДЫГ VS CODE-Д COPY ХИЙНЭ:")
print("   (plate-detector root)")
print("   • main.py")
print("\n3️⃣  DEPENDENCIES СУУЛГАХ:")
print("   pip install --user -r requirements.txt")
print("\n" + "=" * 70 + "\n")
