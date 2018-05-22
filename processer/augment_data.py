import os
import random
from PIL import Image
import numpy as np
import tensorlayer as tl

def augment(file_path):
	print("Processing " + file_path)
	file_names = os.listdir(file_path)
	file_num = len(file_names)
	for i, name in enumerate(file_names):
		if ((i + file_num) >= 30):
			break
		img_path = file_path + name
		img = np.asarray(Image.open(img_path))
		if (random.random() < 0.5):
			new_img = tl.prepro.shift(img, is_random = True)
		else:
			new_img = tl.prepro.rotation(img, is_random = True)
		tl.vis.save_image(new_img, file_path + 'new_%d.jpg' % i)
	print("Done " + file_path)
	return