import cv2
import numpy as np

# 동영상 파일 열기
cap = cv2.VideoCapture(0)

# 화살표 색 범위
lower_black = np.array([0, 0, 0])
upper_black = np.array([179, 255, 30])

while True:
    # 영상에서 프레임 읽어오기
    ret, frame = cap.read()
    if not ret:
        break

    # 색 범위에 따른 이진화
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # 모폴로지 연산을 통한 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 외곽선 검출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 외곽선 중에서 화살표를 검출
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000:
            continue

        # 외곽선 근사화
        approx = cv2.approxPolyDP(contour, cv2.arcLength(contour, True) * 0.02, True)

        # 화살표 모양 판별
        if len(approx) == 7:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / h
            if aspect_ratio < 0.9 or aspect_ratio > 1.1:
                continue

            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)

    # 결과 영상 출력
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# 동영상 파일 닫기
cap.release()
cv2.destroyAllWindows()

