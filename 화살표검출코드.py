#화살표 검출이 가장 깔끔하게 되는 코드 
#-> Canny Edge Detection을 이용해서 그레이스케일, 블러처리 이후 엣지 검출, 검출된 엣지로 윤곽선 검출 -> 화살표 모양 검출
import cv2
import numpy as np

# 비디오 파일 열기
cap = cv2.VideoCapture(0)

# Canny 엣지 검출에 사용될 매개변수 설정
# 찍은 영상에서는 매개변수 값에 큰 영향없이 검출 가능 -> 로봇의 시야에서 보이는 최대 화살표의 크기에 따라 조정하면 될듯
canny_min = 20
canny_max = 120

while True:
    # 비디오에서 한 프레임 읽기
    ret, frame = cap.read()

    if ret:
        # 프레임을 그레이스케일로 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 노이즈를 줄이기 위해 가우시안 블러 적용
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny 엣지 검출 적용
        edges = cv2.Canny(blur, canny_min, canny_max)

        # 윤곽선 찾기
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 찾은 모든 윤곽선에 대해 반복
        for cnt in contours:
            # 윤곽선의 면적 구하기
            area = cv2.contourArea(cnt)

            # 면적이 너무 작으면 무시
            if area < 1000:
                continue

            # 윤곽선의 둘레 구하기
            perimeter = cv2.arcLength(cnt, True)

            # 윤곽선을 다각형으로 근사화
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

            # 다각형이 7개의 꼭지점을 가지면 화살표 모양일 가능성이 높음
            if len(approx) == 7:
                # 윤곽선 주변에 녹색 다각형 그리기
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

        # 프레임 보여주기
        cv2.imshow('Arrow Detection', frame)

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

# 비디오 파일 해제하고 모든 창 닫기
cap.release()
cv2.destroyAllWindows()

#Canny edge detection을 적용하여 경계선을 추출한 다음, contour detection을 적용하여 윤곽선을 찾습니다. 
#그리고 해당 윤곽선을 근사한 다각형이 7개의 꼭지점을 가지고 있다면, 화살표로 판단하여 초록색 다각형을 그려줍니다.
#또한 검출된 화살표의 크기가 일정 이상일 때만 검출되도록 제한을 둡니다.
