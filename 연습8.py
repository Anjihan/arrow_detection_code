import cv2
import numpy as np

# 영상 불러오기
cap = cv2.VideoCapture("example/KakaoTalk_20230506_163340560.mp4")

# 동영상 파일의 프레임 사이즈와 프레임 레이트 계산
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))), (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
frame_rate = cap.get(cv2.CAP_PROP_FPS)

# 동영상 저장을 위한 객체 생성
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, frame_rate, frame_size)

while True:
    # 한 프레임씩 읽어오기
    ret, frame = cap.read()
    if not ret:
        break
    
    # 그레이 스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 블러 처리
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 가장자리 검출
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    
    # 확률적 허프 변환 적용
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
    
    # 직선 그리기 및 화살표 각도 출력
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = -1 * cv2.fastAtan2(float(y2 - y1), float(x2 - x1))
            if 135 <= angle <= 180 or -180 <= angle <= -135:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                print("Arrow angle: ", angle)
    
    # 영상 저장
    out.write(frame)
    
    # 프레임 출력
    cv2.imshow("frame", frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) == ord('q'):
        break

# 객체 해제
cap.release()
out.release()
cv2.destroyAllWindows()