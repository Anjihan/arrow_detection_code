import cv2
import numpy as np

# 비디오 파일 열기
cap = cv2.VideoCapture('example/KakaoTalk_20230506_163340560.mp4')

# Canny 에지 검출을 위한 파라미터 설정
canny_min = 50
canny_max = 150

while True:
    #  비디오에서 프레임 읽기
    ret, frame = cap.read()

    if ret:
        # 프레임을 그레이스케일로 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 노이즈를 줄이기 위해 가우시안 블러 적용
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny 에지 검출 적용
        edges = cv2.Canny(blur, canny_min, canny_max)

        # 윤곽선 검출
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 검출된 모든 윤곽선에 대해 반복
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

            # 다각형의 꼭짓점이 7개이면 화살표 모양일 가능성이 높음
            if len(approx) == 7:
                # 윤곽선 주변에 녹색 다각형 그리기
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

                # 화살표의 각 점의 좌표 구하기
                for i in range(len(approx)):
                    x, y = approx[i][0]
                    #  각 점에 빨간색 원 그리기
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                    # 원 옆에 각 점의 좌표 출력
                    cv2.putText(frame, f"{x}, {y}", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 프레임 보여주기
        cv2.imshow('Arrow Detection', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

# 비디오 파일 해제 및 모든 창 닫기
cap.release()
cv2.destroyAllWindows()
