#완벽한 화살표만 특정 거리에서 검출되는 예제

import cv2
import numpy as np

# 영상 불러오기
cap = cv2.VideoCapture("example\KakaoTalk_20230506_163340560.mp4")

# HSV 색 공간에서 검은색 범위 설정
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 30])

while True:
    # 프레임 읽어오기
    ret, frame = cap.read()
    if not ret:
        break

    # 영상 크기 조절
    frame = cv2.resize(frame, dsize=(640, 480))

    # 색 공간 변환 (BGR -> HSV)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 마스크 생성
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # 노이즈 제거
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 윤곽선 검출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 검출된 윤곽선에서 화살표 모양 찾기
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 7:
            area = cv2.contourArea(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            if 0.8 <= w/h <= 1.2 and area > 3000:
                cv2.drawContours(frame, [cnt], 0, (0, 0, 255), 3)
                
    # 영상 출력
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# 종료
cap.release()
cv2.destroyAllWindows()