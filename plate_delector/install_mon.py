"""
Монгол Tesseract traineddata (mon.traineddata) суулгах скрипт
"""
import os
import sys
import urllib.request
import shutil

# Windows console encoding fix
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

def find_tessdata_folder():
    """Tesseract tessdata хавтсыг олох"""
    # Windows default path
    default_path = r"C:\Program Files\Tesseract-OCR\tessdata"
    
    if os.path.exists(default_path):
        return default_path
    
    # TESSDATA_PREFIX environment variable
    if 'TESSDATA_PREFIX' in os.environ:
        tessdata_path = os.path.join(os.environ['TESSDATA_PREFIX'], 'tessdata')
        if os.path.exists(tessdata_path):
            return tessdata_path
    
    # Try to find tesseract.exe and get tessdata folder
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    
    for tesseract_exe in tesseract_paths:
        if os.path.exists(tesseract_exe):
            tessdata = os.path.join(os.path.dirname(tesseract_exe), 'tessdata')
            if os.path.exists(tessdata):
                return tessdata
    
    return None

def download_mon_traineddata():
    """mon.traineddata файлыг татаж авах"""
    url = "https://raw.githubusercontent.com/tesseract-ocr/tessdata/main/mon.traineddata"
    temp_file = os.path.join(os.environ.get('TEMP', '.'), 'mon.traineddata')
    
    print(f"📥 Татаж байна: {url}")
    try:
        urllib.request.urlretrieve(url, temp_file)
        print(f"✅ Татаж дууслаа: {temp_file}")
        return temp_file
    except Exception as e:
        print(f"❌ Татахад алдаа гарлаа: {e}")
        return None

def install_mon_traineddata(tessdata_folder, source_file):
    """mon.traineddata файлыг tessdata хавтас руу хуулах"""
    dest_file = os.path.join(tessdata_folder, 'mon.traineddata')
    
    try:
        # Хэрэв файл байгаа бол backup хийх
        if os.path.exists(dest_file):
            backup_file = dest_file + '.backup'
            shutil.copy2(dest_file, backup_file)
            print(f"💾 Backup хийсэн: {backup_file}")
        
        # Файлыг хуулах
        shutil.copy2(source_file, dest_file)
        print(f"✅ Суулгасан: {dest_file}")
        return True
    except PermissionError:
        print(f"❌ Эрх хүрэхгүй байна! Administrator эрхтэйгээр ажиллуулна уу.")
        print(f"   Эсвэл файлыг гараар хуулна уу:")
        print(f"   {source_file}")
        print(f"   -> {dest_file}")
        return False
    except Exception as e:
        print(f"❌ Хуулахад алдаа гарлаа: {e}")
        return False

def main():
    print("="*70)
    print(" " * 15 + "MON.TRAINEDDATA СУУЛГАХ")
    print("="*70)
    print()
    
    # Tessdata хавтсыг олох
    print("🔍 Tesseract tessdata хавтсыг хайж байна...")
    tessdata_folder = find_tessdata_folder()
    
    if not tessdata_folder:
        print("❌ Tesseract tessdata хавтас олдсонгүй!")
        print()
        print("💡 Шийдэл:")
        print("   1. Tesseract OCR суулгасан эсэхийг шалгана уу")
        print("   2. Эсвэл tessdata хавтсын замыг гараар оруулна уу:")
        print()
        custom_path = input("   Tessdata хавтсын зам (Enter = цуцлах): ").strip()
        if custom_path and os.path.exists(custom_path):
            tessdata_folder = custom_path
        else:
            print("❌ Цуцлагдлаа.")
            return
    else:
        print(f"✅ Олдлоо: {tessdata_folder}")
    
    print()
    
    # Файлыг татаж авах
    temp_file = download_mon_traineddata()
    if not temp_file:
        return
    
    print()
    
    # Суулгах
    print(f"📦 Суулгаж байна...")
    success = install_mon_traineddata(tessdata_folder, temp_file)
    
    # Түр файлыг устгах
    try:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    except:
        pass
    
    print()
    
    if success:
        print("="*70)
        print("✅ АМЖИЛТТАЙ! mon.traineddata суулгагдлаа!")
        print("="*70)
        print()
        print("💡 Одоо car_plate.py скриптыг дахин ажиллуулна уу.")
        print()
        
        # Шалгах
        mon_file = os.path.join(tessdata_folder, 'mon.traineddata')
        if os.path.exists(mon_file):
            file_size = os.path.getsize(mon_file) / (1024 * 1024)  # MB
            print(f"📊 Файлын хэмжээ: {file_size:.2f} MB")
    else:
        print("="*70)
        print("❌ СУУЛГАХ АМЖИЛТГҮЙ")
        print("="*70)
        print()
        print("💡 Administrator эрхтэйгээр PowerShell ажиллуулж оролдоно уу:")
        print(f"   cd {os.path.dirname(os.path.abspath(__file__))}")
        print("   python install_mon.py")

if __name__ == "__main__":
    main()

