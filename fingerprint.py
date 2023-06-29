
import numpy as np
import cv2
from matplotlib import pyplot as plt
import subprocess
from gtts import gTTS
import os


print("Library Installed")
print('opencv2 version ', cv2.__version__)


max_val = 8
max_pt = -1
max_kp = 0

orb = cv2.ORB_create()
# orb is an alternative to SIFT


test_img = cv2.imread( '101_1.tif', 1 )
cv2.imshow( "Input Image", test_img )

Input_GrayScale = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image", Input_GrayScale)

ret,thresh_binary = cv2.threshold(Input_GrayScale,127,255,cv2.THRESH_BINARY)
ret,thresh_binary_inv = cv2.threshold(Input_GrayScale,127,255,cv2.THRESH_BINARY_INV)
ret,thresh_trunc = cv2.threshold(Input_GrayScale,127,255,cv2.THRESH_TRUNC)
ret,thresh_tozero = cv2.threshold(Input_GrayScale,127,255,cv2.THRESH_TOZERO)
ret,thresh_tozero_inv = cv2.threshold(Input_GrayScale,127,255,cv2.THRESH_TOZERO_INV)

#DISPLAYING THE DIFFERENT THRESHOLDING STYLES
names = ['Oiriginal Image','BINARY','THRESH_BINARY_INV','THRESH_TRUNC','THRESH_TOZERO','THRESH_TOZERO_INV']
images = Input_GrayScale,thresh_binary,thresh_binary_inv,thresh_trunc,thresh_tozero,thresh_tozero_inv

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(names[i])
    plt.xticks([]),plt.yticks([])

plt.savefig('Image Threshold')
plt.show()



# keypoints and descriptors
# (kp1, des1) = orb.detectAndCompute(test_img, None)
(kp1, des1) = orb.detectAndCompute(test_img, None)


#DatasetPath
#training_set = os.path.join('images new/Train/*.jpg')
#print(training_set)
#print(len(training_set))
training_set = ['database/101_1.tif', 'database/101_2.tif', 'database/102_1.tif', 'database/102_2.tif']

#for i in DatasetPath
#        training_set = 

for i in range(0, len(training_set)):
	# train image
	train_img = cv2.imread(training_set[i])

	(kp2, des2) = orb.detectAndCompute(train_img, None)

	# brute force matcher
	bf = cv2.BFMatcher()
	all_matches = bf.knnMatch(des1, des2, k=2)

	good = []
	# give an arbitrary number -> 0.789
	# if good -> append to list of good matches
	for (m, n) in all_matches:
		if m.distance < 0.789 * n.distance:
			good.append([m])

	if len(good) > max_val:
		max_val = len(good)
		max_pt = i
		max_kp = kp2

	print(i, ' ', training_set[i], ' ', len(good))

if max_val != 8:
	print(training_set[max_pt])
	print('good matches ', max_val)

	train_img = cv2.imread(training_set[max_pt])
	img3 = cv2.drawMatchesKnn(test_img, kp1, train_img, max_kp, good, 4)
	
	note = str(training_set[max_pt])[6:-4]
	print('\nFingerprint Matched : ', note)
	plt.imshow(img3)
	plt.savefig( 'Image Keypoints Matched' )
	plt.show()

else:
	print('No Matches')


cv2.destroyAllWindows()
print("Project End")


