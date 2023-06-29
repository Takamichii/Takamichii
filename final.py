from fileinput import filename
import os
from random import sample
from unittest import result
import cv2
from cv2 import SIFT
from cv2 import KeyPoint

sample = cv2.imread( r"C:\Users\USER\Desktop\Fingerprint\Real\1__M_Left_little_finger.BMP")

filename = None
image = None
kp1, kp2, mp = None,None, None
for file in [file for file in os.listdir("Real")][:1000]:
    fingerprint_image = cv2.imread("Fingerprint\Real" + file)
    SIFT = cv2.SIFT_create()

    Keypoints_1 , descrriptors_1 = SIFT.DetectAndComput(sample, None)
    Keypoints_2 , descrriptors_2 = SIFT.detectAndComput(fingerprint_image, None)

    matches = cv2.FlannBasedMatcher ({'algorithm': 1, 'trees': 10},{}).knnmatch(descrriptors_1, descrriptors_2, k=2)

    match_points = []

    for p, q in matches:
        if p.distance < 0.1* q.distance:
            match_points.append(p)

        keypoints = 0 
        if len(keypoints_1) < (keypoints_2):
            keypoints = len(KeyPoint_1)
        else:
            keypoints = len(Keypoints_2)

        if len(match_points) / keypoints * 100 > best_score:
            best_score = len(match_points) / keypoints * 100
            filename = file
            image = fingerprint_image
            kp1, kp2 , mp = Keypoints_1, keypoints_2 , match_points


print("best match: " + filename)
print("SCORE :" + str(best_score))

result = cv2.drawMatches(sample, kp1,image, kp2, mp, None)
result = cv2.resize(result, fx=4, fy=4)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows