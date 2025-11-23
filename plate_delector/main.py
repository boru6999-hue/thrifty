# main.py
import cv2
import os
import sys
from tkinter import Tk, filedialog
from pathlib import Path

from src.detector import FastPlateDetector
from src.utils import format_video_time, put_text_cyrillic
from src.config import Config
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))


def select_video():
    """Видео файл сонгох"""
    root = Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Видео сонгох",
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
            ("All files", "*.*")
        ]
    )
    root.destroy()
    return video_path


def draw_table(frame, detector, video_fps):
    """Хүснэгт зурах"""
    h, w = frame.shape[:2]
    table_w = 380
    table_x = w - table_w

    cv2.rectangle(frame, (table_x, 0), (w, h), (18, 18, 18), -1)
    cv2.rectangle(frame, (table_x, 0), (w, 55), (0, 100, 0), -1)
    cv2.putText(frame, "TANISAN DUGAARUD",
                (table_x + 60, 37), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

    y = 75
    cv2.line(frame, (table_x + 10, y), (w - 10, y), (80, 80, 80), 2)
    y += 28

    cv2.putText(frame, "#", (table_x + 15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    cv2.putText(frame, "Video Tsag", (table_x + 50, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    cv2.putText(frame, "Dugaar", (table_x + 160, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    cv2.putText(frame, "%", (table_x + 320, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    y += 8
    cv2.line(frame, (table_x + 10, y), (w - 10, y), (60, 60, 60), 1)

    detector.clickable_regions = {}
    most_plate, most_count = detector.get_most_detected_plate()
    detector.most_detected_plate = most_plate
    detector.most_detected_count = most_count

    start_idx = max(0, len(detector.detected_plates) - 9)
    for i, det in enumerate(detector.detected_plates[start_idx:], start=start_idx+1):
        y += 38
        if y > h - 80:
            break

        time_str = format_video_time(det['video_time'])
        plate = det['plate']
        conf = det['confidence']

        if plate == most_plate and most_count > 1:
            color = (0, 255, 255)
            cv2.rectangle(frame, (table_x + 5, y - 25),
                          (w - 5, y + 10), (0, 255, 255), 2)
            cv2.putText(frame, "HAMGIIN OLON", (table_x + 10, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        else:
            color = (0, 255, 0) if conf >= 75 else (0, 220, 220)

        cv2.putText(frame, f"{i}", (table_x + 15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(frame, time_str, (table_x + 50, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        plate_x = table_x + 160
        plate_y = y - 20
        plate_w = 200
        plate_h = 30

        if plate in detector.plate_to_file:
            cv2.rectangle(frame, (plate_x - 5, plate_y), (plate_x +
                          plate_w, plate_y + plate_h), (100, 150, 255), 1)
            detector.clickable_regions[(
                plate_x - 5, plate_y, plate_x + plate_w, plate_y + plate_h)] = plate

        put_text_cyrillic(frame, plate, (plate_x, y),
                          font_scale=0.7, color=color, thickness=2)
        cv2.putText(frame, f"{conf:.0f}", (table_x + 320, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    cv2.rectangle(frame, (table_x, h-60), (w, h), (28, 28, 28), -1)
    cv2.putText(frame, f"Niit olson: {len(detector.detected_plates)}",
                (table_x + 20, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)

    unique = len(set(d['plate'] for d in detector.detected_plates))
    cv2.putText(frame, f"Unique: {unique}",
                (table_x + 20, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)

    if detector.most_detected_plate and detector.most_detected_count > 1:
        info_y = h - 200
        cv2.putText(frame, f"HAMGIIN OLON: {detector.most_detected_plate}",
                    (table_x + 20, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"({detector.most_detected_count} udaa)",
                    (table_x + 20, info_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    return frame


def main():
    print("\n" + "="*70)
    print(" "*10 + "🚗 ВИДЕО ДУГААР ТАНИХ (МОНГОЛ ҮСЭГ) 🚗")
    print("="*70)
    print("\n✨ Сайжруулалтууд:")
    print("  • ⭐ МОНГОЛ КИРИЛЛ ҮСЭГ ТАНИЛТ")
    print("  • Хурдан ажиллана (10 frame skip)")
    print("  • Давхцал сайн шалгана (60 frame gap)")
    print("  • Буруу танилт засна")
    print("  • Видеоны цаг харуулна (MM:SS)")
    print("  • Unique дугаар тоолно")
    print("\n" + "-"*70 + "\n")

    detector = FastPlateDetector()

    print(f"\n📋 Тохиргоо:")
    print(
        f"   OCR хэл: {detector.ocr_handler.ocr_lang if detector.ocr_handler.ocr_lang else 'ОЛДСОНГҮЙ!'}")
    print(f"   MIN_CONFIDENCE: {detector.MIN_CONFIDENCE}%")
    print(f"   Debug mode: {'ON' if detector.debug_mode else 'OFF'}\n")

    print("📁 Видео сонгох...")
    video_path = select_video()

    if not video_path:
        print("❌ Видео сонгогдсонгүй!")
        return

    print(f"✅ Видео: {os.path.basename(video_path)}\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Видео нээгдсэнгүй!")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps

    print(f"📊 Мэдээлэл:")
    print(f"   FPS: {fps:.1f}")
    print(f"   Resolution: {width}x{height}")
    print(f"   Frames: {total_frames}")
    print(f"   Үргэлжлэх: {format_video_time(duration)}\n")

    display_w = min(width, Config.DISPLAY_MAX_WIDTH)
    display_h = int(height * (display_w / width))

    print("🚀 Эхэлж байна...\n")
    print("   SPACE - Зогсоох/Үргэлжлүүлэх")
    print("   Q - Дуусгах\n")
    print("-"*70 + "\n")

    frame_count = 0
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n✅ Видео дууслаа!")
                break

            frame_count += 1
            video_time = frame_count / fps

            if frame.shape[1] != display_w:
                frame = cv2.resize(frame, (display_w, display_h))

            if frame_count % 10 == 0:
                plates = detector.detect_plates(frame)

                for (x, y, w, h) in plates:
                    plate_img = frame[y:y+h, x:x+w]

                    if plate_img.size == 0:
                        continue

                    enhanced = detector.enhance_plate_fast(plate_img)
                    text, conf = detector.ocr_plate(enhanced)

                    if text:
                        if not detector.is_duplicate(text, frame_count, video_time):
                            detector.last_plate_image = plate_img.copy()
                            detector.last_plate_text = text
                            detector.last_plate_conf = conf

                            detector.detected_plates.append({
                                'plate': text,
                                'confidence': conf,
                                'video_time': video_time,
                                'frame': frame_count
                            })

                            most_plate, most_count = detector.get_most_detected_plate()

                            if most_count > detector.most_detected_count:
                                detector.most_detected_plate = most_plate
                                detector.most_detected_count = most_count
                                print(
                                    f"⭐ ХАМГИЙН ОЛОН: {most_plate} ({most_count} удаа)")

                                if text == most_plate:
                                    detector.file_handler.save_result(
                                        plate_img, text, video_time)

                            time_str = format_video_time(video_time)
                            print(
                                f"✅ {len(detector.detected_plates)}. {text} ({conf:.0f}%) @ {time_str}")

                        if conf >= detector.MIN_CONFIDENCE:
                            frame = detector.draw_detection(
                                frame, x, y, w, h, text, conf)
                    else:
                        cv2.rectangle(frame, (x, y), (x+w, y+h),
                                      (0, 0, 255), 2)
                        cv2.putText(frame, "OCR failed", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            frame = draw_table(frame, detector, fps)
            frame = detector.draw_plate_preview(frame)

            h_frame = frame.shape[0]
            cv2.rectangle(frame, (5, h_frame-50),
                          (350, h_frame-5), (25, 25, 25), -1)

            curr_time_str = format_video_time(video_time)
            total_time_str = format_video_time(duration)
            cv2.putText(frame, f"Video: {curr_time_str} / {total_time_str}",
                        (10, h_frame-28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)

            progress = (frame_count / total_frames) * 100
            cv2.putText(frame, f"Progress: {progress:.1f}%",
                        (10, h_frame-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

        cv2.imshow('Video Plate Detection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord(' '):
            paused = not paused
            print("⏸️  Зогсоосон" if paused else "▶️  Үргэлжилж байна")

    cap.release()
    cv2.destroyAllWindows()

    # Дүгнэлт
    print("\n" + "="*70)
    print(" "*25 + "📊 ДҮГНЭЛТ")
    print("="*70)
    print(f"Нийт танигдсан: {len(detector.detected_plates)}")

    if detector.detected_plates:
        unique_plates = set(d['plate'] for d in detector.detected_plates)
        print(f"Unique дугаар: {len(unique_plates)}\n")

        avg_conf = sum(d['confidence']
                       for d in detector.detected_plates) / len(detector.detected_plates)
        print(f"Дундаж confidence: {avg_conf:.1f}%")

    print(f"\n💾 Хадгалагдсан файлууд: {Config.DETECTED_PLATES_DIR}")
    print("\n👋 Баяртай!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
