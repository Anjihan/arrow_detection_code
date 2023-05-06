import cv2
import numpy as np

# 화살표 감지 함수
def detect_arrow(frame):
    # 그레이 스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 가우시안 블러 필터
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 캐니 엣지 검출
    edged = cv2.Canny(blurred, 30, 150)

    # 허프 변환을 이용한 직선 검출
    lines = cv2.HoughLinesP(edged, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)

    # 직선이 검출되면
    if lines is not None:
        # 화살표가 아닌 다른 선들은 제외하고 화살표만 검출
        arrow_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            angle = np.arctan2(dy, dx) * 180 / np.pi
            if angle > -45 and angle < 45 and dx > 0:
                arrow_lines.append(line)
        # 검출된 화살표 선 그리기
        for line in arrow_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # 검출된 화살표 방향 벡터 출력
        if len(arrow_lines) > 0:
            x1, y1, x2, y2 = arrow_lines[0][0]
            dx, dy = x2 - x1, y2 - y1
            angle = np.arctan2(dy, dx) * 180 / np.pi
            if angle < 0:
                angle += 360
            direction = round(angle / 45) % 8
            return direction
    
    return None

# 비디오 캡쳐 객체 생성
cap = cv2.VideoCapture('example/KakaoTalk_20230506_163340560.mp4')

# 비디오 프레임 반복 처리
while cap.isOpened():
    # 비디오 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 화살표 검출 함수 호출
    direction = detect_arrow(frame)
    
    # 검출된 화살표 방향 출력
    if direction is not None:
        cv2.putText(frame, str(direction), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    
    # 프레임 출력
    cv2.imshow('frame', frame)
    
    # q 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오

# 자원 해제
cap.release()
cv2.destroyAllWindows()
