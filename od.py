import cv2

print("Камерууд хайж байна...")

# 0, 1, 2 гэх мэт янз бүрийн камер турших
for i in range(5):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if cap.isOpened():
        print(f"✅ Камер #{i} олдлоо!")
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f'Camera {i}', frame)
            cv2.waitKey(2000)  # 2 секунд харуулна
        cap.release()
    else:
        print(f"❌ Камер #{i} олдсонгүй")

cv2.destroyAllWindows()
print("\nДууслаа!")