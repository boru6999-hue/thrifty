import cv2
import pytesseract
import numpy as np

# ⚠️ Windows хэрэглэгч: доорх замыг өөрийнхөөр солино
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def preprocess_plate(plate_img):
    """Дугаарын зургийг боловсруулах: grayscale → threshold → морфологи"""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    # Thresholding (улаан/цайвар дугаарын хувьд сайн)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Морфологийн арилга (noise багасгах)
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return clean


def detect_and_read_plate(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Алдаа: Зургийг уншиж чадсангүй. Замыг шалгана уу.")
        return None

    # 1. Хүрээ илрүүлэх (хялбар хувилбар: контурын суурьтай)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(
        edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    plate = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        # Дөрвөлжин/тэгш өнцөгт хэлбэртэй контурыг сонгоно
        if len(approx) == 4:
            plate = approx
            break

    if plate is None:
        print("⚠️ Дугаарын хавтан олдсонгүй. OCR-г шууд бүх зураг дээр ажиллуулъя...")
        cropped = img
    else:
        # Дугаарын хэсгийг тусдаа авах (perspective transform)
        pts = plate.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]      # Зүүн дээд
        rect[2] = pts[np.argmax(s)]      # Баруун доод
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]   # Баруун дээд
        rect[3] = pts[np.argmax(diff)]   # Зүүн доод

        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        cropped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # 2. OCR хийх
    processed = preprocess_plate(cropped)

    # OCR параметр: монгол дугаарт тохирсон (өндөр, өргөн, тоо/үсэг)
    custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789АБВГДЕЁЖЗИКЛМНОӨПРСТУҮФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмноөпрстуүфхцчшщъыьэюя'
    # Монгол дэмжихгүй тул 'eng' + whitelist
    text = pytesseract.image_to_string(
        processed, config=custom_config, lang='eng')

    # Цэвэрлэх
    text = ''.join(ch for ch in text if ch.isalnum()).upper()
    return text


# 🚀 Жишээ ашиглалт
if __name__ == "__main__":
    image_path = "images.jpeg"  # ← Энд өөрийн зургийн замыг оруулна
    result = detect_and_read_plate(image_path)
    if result:
        print(f"✅ Танигдсан дугаар: **{result}**")
    else:
        print("❌ Дугаар танигдаагүй.")
