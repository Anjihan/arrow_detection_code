#빨간색 윤곽선으로 화살표 검출
import cv2
import numpy as np

# 영상 파일 열기
cap = cv2.VideoCapture('example/KakaoTalk_20230506_163340560.mp4')

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break
    
    # 그레이 스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Canny Edge Detection 수행
    edges = cv2.Canny(gray, 50, 200, apertureSize=3)
    
    # Contour Detection 수행
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Contour 중에서 화살표 모양 검출하기
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02*perimeter, True)
        
        if len(approx) == 7: # 7개의 꼭지점이 있는 contour가 화살표 모양일 가능성이 있다
            x,y,w,h = cv2.boundingRect(contour)
            cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)
    
    # 결과 출력
    cv2.imshow('Arrow Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# 해제
cap.release()
cv2.destroyAllWindows()
