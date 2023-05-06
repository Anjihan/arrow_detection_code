#Canny edge detection을 적용하여 경계선을 추출한 다음, contour detection을 적용하여 윤곽선을 찾습니다. 
#그리고 해당 윤곽선을 근사한 다각형이 7개의 꼭지점을 가지고 있다면, 화살표로 판단하여 초록색 다각형을 그려줍니다.
#또한 검출된 화살표의 크기가 일정 이상일 때만 검출되도록 제한을 둡니다.
import cv2
import numpy as np

# Open the video file
cap = cv2.VideoCapture('example/KakaoTalk_20230506_163340560.mp4')

# Set up the parameters for Canny edge detection
canny_min = 50
canny_max = 150

while True:
    # Read a frame from the video
    ret, frame = cap.read()

    if ret:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blur, canny_min, canny_max)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate through all the contours found
        for cnt in contours:
            # Get the area of the contour
            area = cv2.contourArea(cnt)

            # If the area is too small, ignore it
            if area < 1000:
                continue

            # Get the perimeter of the contour
            perimeter = cv2.arcLength(cnt, True)

            # Approximate the contour to a polygon
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)

            # If the polygon has 7 vertices, it is likely the arrow shape we are looking for
            if len(approx) == 7:
                # Draw a green polygon around the contour
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow('Arrow Detection', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

# Release the video file and close all windows
cap.release()
cv2.destroyAllWindows()
