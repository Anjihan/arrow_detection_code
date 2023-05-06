import cv2
import numpy as np

cap = cv2.VideoCapture("example/KakaoTalk_20230506_163340560.mp4")

while True:
    # 프레임 읽기
    ret, frame = cap.read()

    # 동영상 끝나면 종료
    if not ret:
        break

    # 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 블러 처리
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 캐니 에지 검출
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)

    # 컨투어 검출
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 컨투어 전처리
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        # 컨투어 근사화
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)

        # 꼭지점 개수 확인
        if len(approx) != 7:
            continue

        # 컨투어 그리기
        cv2.drawContours(frame, [cnt], 0, (0, 255, 0), 3)

        # 중심점 계산
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # 각도 계산
            pts = approx.reshape(7, 2)
            pts = pts[np.argsort(pts[:, 1])]
            p1, p2, p3, p4, p5, p6, p7 = pts

            dx1 = p4[0] - p7[0]
            dy1 = p4[1] - p7[1]
            dx2 = p1[0] - p7[0]
            dy2 = p1[1] - p7[1]

            angle = np.arctan2(dy1+dy2, dx1+dx2) * 180 / np.pi
            print(f"Arrow angle: {angle:.2f}")
            #arrow_angle은 화살표가 수평선에서 시계 방향으로 회전한 각도를 나타냄
            #ex> 0도는 수평선과 평행한 화살표를 의미하며, 90도는 수직선과 평행한 화살표를 의미

    # 출력 이미지 크기 조정
    scale_percent = 520 / frame.shape[0]
    width = int(frame.shape[1] * scale_percent)
    height = int(frame.shape[0] * scale_percent)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

    # 결과 출력
    cv2.imshow('Arrow Detection', resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
cap.release()
cv2.destroyAllWindows()
