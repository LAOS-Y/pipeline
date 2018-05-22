import numpy as np
from models.kmeans import KMeans
from models.sift import genSiftFeatures


class VBoW(object):
	def __init__(self):
		pass
	
	def genWords(self, X, K):
		N = X.shape[0]
		self.sift_features = genSiftFeatures(X)
		features_list = []
		for i in range(N):
			features_list = features_list + self.sift_features[i]
		print(np.array(features_list).shape)
		model = KMeans()
		self.Words = model.train(np.array(features_list), K)
		return self.Words
	
	def genFeatures(self, X):
		N = X.shape[0]
		D = self.Words.shape[0]
		y = np.zeros([N, D])
		model = KMeans()
		model.clusters = self.Words
		for i in range(N):
			img_words = model.predict(np.array(self.sift_features[i]))
			y[i] = np.bincount(np.array(img_words), minlength = D)
		return y