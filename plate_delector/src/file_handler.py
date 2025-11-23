import os
import cv2
import subprocess
import platform
from datetime import datetime
from .config import Config


class FileHandler:
    """Файл удирдлагын класс"""

    def __init__(self):
        self.save_folder = Config.DETECTED_PLATES_DIR
        self.plate_to_file = {}  # {plate_text: file_path}

        # Хавтас үүсгэх
        self._ensure_directories()

        # Одоо байгаа файлуудыг ачаалах
        self._load_existing_files()

    def _ensure_directories(self):
        """Шаардлагатай хавтсуудыг үүсгэх"""
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def _load_existing_files(self):
        """Одоо байгаа хадгалагдсан файлуудыг ачаалах"""
        try:
            if os.path.exists(self.save_folder):
                for filename in os.listdir(self.save_folder):
                    if filename.endswith('.jpg') and not filename.startswith('_LOW_'):
                        parts = filename.replace('.jpg', '').split('_')
                        if len(parts) >= 1:
                            plate_text = parts[0]
                            if plate_text and len(plate_text) >= 4:
                                file_path = os.path.join(
                                    self.save_folder, filename)
                                self.plate_to_file[plate_text] = os.path.abspath(
                                    file_path)
        except Exception as e:
            print(f"⚠️  Файл ачаалах алдаа: {e}")

    def format_video_time(self, seconds):
        """Секунд → MM:SS формат"""
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    def save_result(self, plate_img, text, video_time):
        """
        Дугаарын зургийг хадгалах

        Args:
            plate_img: Зургийн numpy array
            text: Дугаарын текст
            video_time: Видеоны цаг (секунд)

        Returns:
            bool: Амжилттай эсэх
        """
        try:
            if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder)

            time_str = self.format_video_time(video_time)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            safe_text = "".join(
                c for c in text if c.isalnum() or c in ('-', '_'))
            filename = f"{safe_text}_{time_str.replace(':', '-')}_{timestamp}.jpg"
            img_file = os.path.join(self.save_folder, filename)

            success = cv2.imwrite(img_file, plate_img)

            if success:
                self.plate_to_file[text] = os.path.abspath(img_file)
                print(f"💾 Хадгалсан: {filename}")
                return True
            else:
                print(f"❌ Хадгалах амжилтгүй: {filename}")
                return False

        except Exception as e:
            print(f"❌ Хадгалах алдаа: {e}")
            return False

    def delete_file(self, plate_text):
        """Дугаарын файлыг устгах"""
        try:
            if plate_text in self.plate_to_file:
                file_path = self.plate_to_file[plate_text]
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"🗑️  Файл устгасан: {os.path.basename(file_path)}")
                    del self.plate_to_file[plate_text]
                    return True
        except Exception as e:
            print(f"❌ Файл устгах алдаа: {e}")
        return False

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

    def get_saved_files_count(self):
        """Хадгалагдсан файлуудын тоо"""
        if os.path.exists(self.save_folder):
            return len([f for f in os.listdir(self.save_folder) if f.endswith('.jpg')])
        return 0

    def list_saved_files(self):
        """Хадгалагдсан файлуудын жагсаалт"""
        files = []
        if os.path.exists(self.save_folder):
            files = [f for f in os.listdir(
                self.save_folder) if f.endswith('.jpg')]
        return files
