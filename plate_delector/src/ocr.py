import cv2
import pytesseract
from .config import Config


class OCRHandler:
    """OCR handler for plate recognition"""

    def __init__(self):
        self.min_confidence = Config.MIN_CONFIDENCE
        self.ocr_lang = self._detect_ocr_language()

        # Debug
        self.debug_mode = Config.DEBUG_MODE
        self.debug_count = 0

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
                print(
                    "   💡 Tesseract-д 'rus.traineddata' эсвэл 'mon.traineddata' суулгана уу.")
                print("   📥 Татаж авах: https://github.com/tesseract-ocr/tessdata")
                return None
        except Exception as e:
            print(f"⚠️  Tesseract хэл шалгах алдаа: {e}")
            return None

    def ocr_improved(self, img):
        """
        OCR - Монгол кирилл үсэг ба цифр танина (mon эсвэл rus хэл)

        Returns:
            tuple: (text, confidence) эсвэл (None, 0)
        """
        if self.ocr_lang is None:
            if self.debug_mode and self.debug_count < 5:
                print("❌ OCR хэл тохируулагдаагүй байна!")
                self.debug_count += 1
            return None, 0

        # Зөвхөн Монгол кирилл том үсгүүд болон цифр
        mongolian_letters = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯӨҮ'
        whitelist = mongolian_letters + '0123456789'

        # Олон PSM режим туршиж үзэх
        psm_modes = [7, 8, 6, 11]
        text = None
        best_text = None
        best_conf = 0

        for psm in psm_modes:
            config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist={whitelist}'
            try:
                text = pytesseract.image_to_string(
                    img, config=config, lang=self.ocr_lang)
                if text and text.strip():
                    try:
                        data = pytesseract.image_to_data(
                            img, config=config, lang=self.ocr_lang,
                            output_type=pytesseract.Output.DICT)
                        confidences = [int(conf)
                                       for conf in data['conf'] if int(conf) > 0]
                        avg_conf = sum(confidences) / \
                            len(confidences) if confidences else 0
                    except:
                        avg_conf = 50

                    if avg_conf > best_conf:
                        best_text = text
                        best_conf = avg_conf
            except Exception as e:
                if self.ocr_lang == 'mon':
                    try:
                        text = pytesseract.image_to_string(
                            img, config=config, lang='rus')
                        if text and text.strip():
                            self.ocr_lang = 'rus'
                            if self.debug_mode and self.debug_count < 3:
                                print(
                                    "⚠️  Монгол хэл амжилтгүй, Орос хэл рүү шилжлээ.")
                                self.debug_count += 1
                            try:
                                data = pytesseract.image_to_data(
                                    img, config=config, lang='rus',
                                    output_type=pytesseract.Output.DICT)
                                confidences = [
                                    int(conf) for conf in data['conf'] if int(conf) > 0]
                                avg_conf = sum(
                                    confidences) / len(confidences) if confidences else 0
                            except:
                                avg_conf = 50
                            if avg_conf > best_conf:
                                best_text = text
                                best_conf = avg_conf
                    except:
                        continue

        if not best_text:
            try:
                config = f'--oem 3 --psm 7'
                text = pytesseract.image_to_string(
                    img, config=config, lang=self.ocr_lang)
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
            print(
                f"🔍 OCR raw: '{best_text}' -> cleaned: '{cleaned}' (conf: {best_conf:.1f})")
            self.debug_count += 1

        if cleaned:
            digit_count = sum(c.isdigit() for c in cleaned)
            letter_count = sum(c.isalpha() for c in cleaned)

            if digit_count == 0 or letter_count == 0:
                balance = 0.0
            else:
                balance = min(digit_count, letter_count) / \
                    max(digit_count, letter_count)

            conf = max(40, min(90, best_conf * 0.8 + (balance * 20)))
            return cleaned, conf

        return None, 0

    def clean_and_fix_text(self, text):
        """
        Текст цэвэрлэх + Монгол дугаарын формат руу засах (4 тоо + 3 үсэг)

        Returns:
            str: Цэвэрлэгдсэн текст (7 тэмдэгт) эсвэл None
        """
        if not text:
            return None

        text = text.strip().upper()

        # Зөвхөн Монгол кирилл том үсэг болон цифр үлдээх
        mongolian_letters = set('АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯӨҮ')
        allowed = mongolian_letters.union(set('0123456789'))

        cleaned = ''.join(c for c in text if c in allowed)

        if len(cleaned) < 6:
            return None

        # Кирилл→цифр засварууд
        corrections_to_digit = {
            'О': '0', 'С': '5', 'З': '3',
            'Б': '6', 'И': '1', 'Л': '1',
        }

        # Цифр→кирилл засварууд
        corrections_to_letter = {
            '0': 'О', '5': 'С', '3': 'З',
            '6': 'Б', '1': 'И',
        }

        digits = []
        letters = []
        ambiguous = []

        for c in cleaned:
            if c.isdigit():
                digits.append(c)
            elif c in mongolian_letters:
                letters.append(c)
            elif c in corrections_to_digit:
                ambiguous.append((c, 'digit'))
            elif c in corrections_to_letter.values():
                ambiguous.append((c, 'letter'))

        for char, target_type in ambiguous:
            if target_type == 'digit' and len(digits) < 4:
                digits.append(corrections_to_digit[char])
            elif target_type == 'letter' and len(letters) < 3:
                if char in corrections_to_letter:
                    letters.append(char)

        if len(digits) < 4 and len(letters) > 3:
            for c in letters[:len(letters)-3]:
                if c in corrections_to_digit:
                    digits.append(corrections_to_digit[c])
                    letters.remove(c)
                    if len(digits) >= 4:
                        break

        if len(letters) < 3 and len(digits) > 4:
            for c in digits[4:]:
                if c in corrections_to_letter:
                    letters.append(corrections_to_letter[c])
                    digits.remove(c)
                    if len(letters) >= 3:
                        break

        first_four = ''.join(digits[:4])
        if len(first_four) < 4:
            return None

        last_three = ''.join(letters[:3])
        if len(last_three) < 3:
            return None

        result = first_four + last_three

        if len(result) != 7:
            return None

        return result
