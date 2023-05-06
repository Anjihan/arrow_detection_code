import cv2
import numpy as np

cap = cv2.VideoCapture("example\KakaoTalk_20230506_163340560.mp4")

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 200, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)
    
    # 화살표 정보를 저장할 리스트
    arrows = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 라인의 길이 구하기
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length < 50:
                continue
            # 라인의 각도 구하기
            angle = -1 * np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            # 라인을 그리기 위한 시작점과 끝점 좌표
            start_point = (x1, y1)
            end_point = (x2, y2)
            # 화살표에 대한 정보를 arrows 리스트에 추가
            arrows.append((start_point, end_point, angle))

        # arrows 리스트를 기준으로 빨간색 선 그리기
        for arrow in arrows:
            start_point, end_point, angle = arrow
            cv2.line(frame, start_point, end_point, (0, 0, 255), 2)
            cv2.putText(frame, str(round(angle, 2)), end_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()