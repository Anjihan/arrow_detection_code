import cv2
import numpy as np

# 영상 파일 읽기
cap = cv2.VideoCapture('example/KakaoTalk_20230506_163340560.mp4')

while True:
    # 영상 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break
    
    # 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Canny 에지 검출
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 선분 검출
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # 화살표 검출
    arrow = None
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # 선분 길이 계산
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # 선분 방향 계산
            angle = -1 * cv2.fastAtan2(float(y2 - y1), float(x2 - x1))
            if angle < 0:
                angle += 360

            # 화살표 검출 조건 확인
            if length > 80 and angle > 50 and angle < 130:
                arrow = line
                break

    # 화살표 그리기
    if arrow is not None:
        x1, y1, x2, y2 = arrow[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 화살표 방향벡터 출력
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        dx = x2 - x1
        dy = y2 - y1
        print(f"Arrow direction vector: ({dx}, {dy})")

        cv2.arrowedLine(frame, (cx, cy), (cx + dx, cy + dy), (0, 255, 0), 2)

    # 영상 출력
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# 종료
cap.release()
cv2.destroyAllWindows()
