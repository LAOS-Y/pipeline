def show_img(img):
	import matplotlib.pyplot as plt
	plt.imshow(img)
	plt.show()

def to_one_hot(values):
	import numpy as np
	return np.eye(values.max() + 1)[values]