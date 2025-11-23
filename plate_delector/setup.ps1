# setup.ps1
# Энэ скриптийг plate-detector хавтас дотор ажиллуулна
# PowerShell дээр: .\setup.ps1

Write-Host "🔧 Folder структур үүсгэж байна..." -ForegroundColor Green

# Folder үүсгэх
New-Item -ItemType Directory -Path "src" -Force -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path "resources\cascades" -Force -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path "output\detected_plates" -Force -ErrorAction SilentlyContinue | Out-Null
New-Item -ItemType Directory -Path "logs" -Force -ErrorAction SilentlyContinue | Out-Null

Write-Host "✅ Папкууд үүсгэгдсэн" -ForegroundColor Green

# requirements.txt үүсгэх
Write-Host "`n📝 requirements.txt үүсгэж байна..." -ForegroundColor Green

$requirements = @"
opencv-python==4.8.0.74
pytesseract==0.3.10
Pillow==10.0.0
numpy==1.24.0
"@

Set-Content -Path "requirements.txt" -Value $requirements -Encoding UTF8
Write-Host "✅ requirements.txt үүсгэгдсэн" -ForegroundColor Green

# src/__init__.py үүсгэх
Write-Host "`n📝 src/__init__.py үүсгэж байна..." -ForegroundColor Green

$init_py = @"
# src/__init__.py
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
"@

Set-Content -Path "src/__init__.py" -Value $init_py -Encoding UTF8
Write-Host "✅ src/__init__.py үүсгэгдсэн" -ForegroundColor Green

Write-Host "`n" 
Write-Host "="*70 -ForegroundColor Cyan
Write-Host "📋 ДАРААГИЙН АЛХАМ" -ForegroundColor Cyan
Write-Host "="*70 -ForegroundColor Cyan

Write-Host "`n1️⃣  Доорх файлуудыг VS Code-д үүсгээнэ (src/ папка дотор):" -ForegroundColor Yellow
Write-Host "   📄 config.py"
Write-Host "   📄 ocr.py"
Write-Host "   📄 utils.py"
Write-Host "   📄 file_handler.py"
Write-Host "   📄 detector.py"

Write-Host "`n2️⃣  Доорх файлуудыг VS Code-д үүсгээнэ (project root):" -ForegroundColor Yellow
Write-Host "   📄 main.py"

Write-Host "`n3️⃣  Doosh dependencies suulgah:" -ForegroundColor Yellow
Write-Host "   pip install --user -r requirements.txt" -ForegroundColor Cyan

Write-Host "`n4️⃣  Tesseract suulgah:" -ForegroundColor Yellow
Write-Host "   https://github.com/UB-Mannheim/tesseract/wiki" -ForegroundColor Cyan

Write-Host "`n5️⃣  Mongol hel suulgah:" -ForegroundColor Yellow
Write-Host "   tessdata/mon.traineddata → C:\Program Files\Tesseract-OCR\tessdata" -ForegroundColor Cyan

Write-Host "`n6️⃣  Ajilluulah:" -ForegroundColor Yellow
Write-Host "   python main.py" -ForegroundColor Cyan

Write-Host "`n" 
Write-Host "="*70 -ForegroundColor Cyan
Write-Host "✅ Бэлэн! VS Code-д файлуудыг үүсгээрэй" -ForegroundColor Green
Write-Host "="*70 -ForegroundColor Cyan
Write-Host ""