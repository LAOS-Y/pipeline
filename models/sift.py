import cv2

def toGray(color_img):
	gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
	return gray_img

def genSiftFeatures(X):
	features = []
	sift = cv2.xfeatures2d.SIFT_create()
	for i in range(X.shape[0]):
		x = toGray(X[i])
		kp, desc = sift.detectAndCompute(x, None)
		features.append(desc.tolist())
	return features