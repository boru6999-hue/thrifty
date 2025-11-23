# setup.py
# Энэ файлыг plate-detector хавтас дотор ажиллуулна
# python setup.py

import os
import sys

print("\n" + "="*70)
print("🔧 PLATE DETECTOR PROJECT SETUP")
print("="*70 + "\n")

# Folder үүсгэх
folders = [
    'src',
    'resources/cascades',
    'output/detected_plates',
    'logs'
]

print("📁 Папкууд үүсгэж байна...\n")
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"   ✅ {folder}")

# requirements.txt
print("\n📝 requirements.txt үүсгэж байна...")
req_content = """opencv-python==4.8.0.74
pytesseract==0.3.10
Pillow==10.0.0
numpy==1.24.0
"""

with open('requirements.txt', 'w', encoding='utf-8') as f:
    f.write(req_content)
print("   ✅ requirements.txt үүсгэгдсэн")

# src/__init__.py
print("\n📝 src/__init__.py үүсгэж байна...")
init_content = """# src/__init__.py
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
"""

with open('src/__init__.py', 'w', encoding='utf-8') as f:
    f.write(init_content)
print("   ✅ src/__init__.py үүсгэгдсэн")

print("\n" + "="*70)
print("📋 ДАРААГИЙН АЛХАМ")
print("="*70)

print("\n1️⃣  Доорх файлуудыг VS Code-д COPY ХИЙНЭ (src/ папка дотор):")
print("   📄 config.py")
print("   📄 ocr.py")
print("   📄 utils.py")
print("   📄 file_handler.py")
print("   📄 detector.py")

print("\n2️⃣  Доорх файлуудыг VS Code-д COPY ХИЙНЭ (project root):")
print("   📄 main.py")

print("\n3️⃣  Dependencies суулгах:")
print("   pip install --user -r requirements.txt")

print("\n4️⃣  Tesseract суулгах (байхгүй бол):")
print("   https://github.com/UB-Mannheim/tesseract/wiki")
print("   Default: C:\\Program Files\\Tesseract-OCR")

print("\n5️⃣  Монгол хэл суулгах:")
print("   tessdata/mon.traineddata → C:\\Program Files\\Tesseract-OCR\\tessdata")

print("\n6️⃣  Ажиллуулах:")
print("   python main.py")

print("\n" + "="*70)
print("✅ SETUP ДУУСАА! Файлуудыг copy хийнэ")
print("="*70 + "\n")
