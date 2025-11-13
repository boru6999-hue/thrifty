import cv2
import numpy as np
import pytesseract
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

print("=" * 60)
print("         МАШИНЫ ДУГААР ТАНИХ СИСТЕМ")
print("=" * 60)
print("\n🎥 Камер асаж байна...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" АЛДАА: Камер нээгдсэнгүй!")
    print("Камераа шалгаад дахин оролдоно уу.")
    exit()

print(" Камер амжилттай нээгдлээ!\n")
print(" SPACE - Зураг авч дугаар таних")
print(" Q - Програмаас гарах")
print("-" * 60)

cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

saved_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print(" Камераас дүрс уншиж чадсангүй!")
        break
    display_frame = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    plates = cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in plates:
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(display_frame, "Dugaar oldloo!", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(display_frame, "SPACE - Zurgar awah | Q - Garah",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    if len(plates) > 0:
        cv2.putText(display_frame, f"{len(plates)} dugaar oldloo!",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Mashinii Dugaar Tanikh', display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        if len(plates) == 0:
            print("\n⚠️  Дугаар олдсонгүй! Дахин оролдоно уу.")
        else:
            print(f"\n{'='*60}")
            print(f" {len(plates)} дугаар илэрлээ!")
            print('='*60)

            for i, (x, y, w, h) in enumerate(plates):
                plate_img = frame[y:y+h, x:x+w]

                gray_plate = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray_plate, (5, 5), 0)
                _, thresh = cv2.threshold(
                    blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                try:
                    config = '--oem 3 --psm 7'
                    text = pytesseract.image_to_string(thresh, config=config)
                    text = text.strip().replace('\n', '').replace(' ', '')

                    if text:
                        print(f"\n🚗 Дугаар #{i+1}: {text}")
                    else:
                        print(f"\ns  Дугаар #{i+1}: Танигдсангүй")
                        text = "Unknown"
                except Exception as e:
                    print(f"\n Алдаа гарлаа: {e}")
                    text = "Error"

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"plate_{timestamp}_{i+1}.jpg"
                cv2.imwrite(filename, plate_img)
                print(f"💾 Хадгалсан файл: {filename}")

                saved_count += 1

            print('='*60)
            print(f" Нийт хадгалсан: {saved_count} зураг\n")

    elif key == ord('q') or key == ord('Q'):
        print("\n" + "="*60)
        print(" Баяртай!")
        print(f" Нийт {saved_count} зураг хадгаллаа.")
        print("="*60)
        break

cap.release()
cv2.destroyAllWindows()
