import cv2
import numpy as np

# 동영상 파일 열기
cap = cv2.VideoCapture("example/KakaoTalk_20230506_163340560.mp4")

while True:
    # 동영상에서 새로운 프레임 읽기
    ret, frame = cap.read()

    if not ret:
        break

    # 그레이스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 블러 처리 및 Canny edge detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Contour detection
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # contour의 둘레 길이 계산
        perimeter = cv2.arcLength(cnt, True)

        # contour를 근사화하여 좌표를 줄임
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

        # contour가 화살표인지 검사
        if len(approx) == 7:
            # contour의 중심점 계산
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # 화살표 시작점과 끝점의 상대 위치 계산
            start = approx[1][0]
            end = approx[5][0]
            dx = end[0] - start[0]
            dy = end[1] - start[1]

            # 방향각 계산
            angle = np.degrees(np.arctan2(dy, dx))
            if angle < 0:
                angle += 180

            # 화살표 윤곽 그리기
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)

            # 방향각 표시
            cv2.putText(frame, f"Angle: {angle:.1f} deg", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 화면 출력
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Arrow Detection", frame)

    # 종료를 위한 키 처리
    if cv2.waitKey(1) == ord('q'):
        break

# 종료
cap.release()
cv2.destroyAllWindows()
