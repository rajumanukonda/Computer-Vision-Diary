import cv2
from cvzone.PoseModule import PoseDetector
cap = cv2.VideoCapture("inputVideo3.mp4")  # Give your video input here , '0' for webcam and file name for video files.

detector = PoseDetector()
posList = []

while True:

    success, img = cap.read()
    img = detector.findPose(img)

    landmarkList, bboxInfo = detector.findPosition(img)     # Returns 33 landmark points (x y z) for each frame

    if bboxInfo:
        landmarkString = ''
        for landmark in landmarkList:
            # Segregating the landmark points for processing
            landmarkString += f'{landmark[1]},{img.shape[0] - landmark[2]},{landmark[3]},'

        posList.append(landmarkString)

    cv2.imshow('', img)
    key = cv2.waitKey(1)

    # Storing the required landmark point on keystroke. Will be helpful in reduction of datapoints.
    if key == ord('s'):
        with open("AnimationFile.txt", "w") as f:
            f.writelines(["%s\n" % item for item in posList])
