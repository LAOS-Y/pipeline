import numpy as np


class KMeans(object):
	def __init__(self):
		pass

	def train(self, X, K):
		N = X.shape[0]
		D = X.shape[1]
		self.clusters = np.zeros([K, D])
		for i in range(K):
			self.clusters[i] = X[i]
		iter = 0
		is_changed = 1
		while is_changed:
			iter += 1
			print("Begin iteration #%d" % (iter))
			indexs = []
			for i in range(K):
				indexs.append([])
			is_changed = 0
			for i in range(N):
				class_id = np.argmin(np.sum((self.clusters - X[i]) ** 2, axis = 1))
				indexs[class_id].append(i)
			for class_id, index in enumerate(indexs):
				new_cluster = np.mean(X[index], axis = 0)
				if ((new_cluster != self.clusters[class_id]).any()):
					self.clusters[class_id] = new_cluster
					is_changed = 1
			print("Done iteration #%d" % (iter))
		return self.clusters
	
	def predict(self, X):
		N = X.shape[0]
		y = []
		for i in range(N):
			y.append(np.argmin(np.sum((self.clusters - X[i]) ** 2, axis = 1)))
		return y