import numpy as np 
from PIL import Image

def load_and_resize(file_labels, file_path, resize_list=[(400, 400), (299, 299), (224, 224)]):
	with open(file_labels) as f:
		img_dir = file_path
		lines = f.readlines()
		img_list = []
		for line in lines:
			temp = line.strip('\n').split(' ')
			img_name = temp[0]
			img_class = int(temp[1]) - 1
			img_list.append([img_name, img_class])
	np.random.seed(42)
	indice_mask = np.random.permutation(len(img_list))
	new_img_list = [[]]
	for _ in resize_list:
		new_img_list.append([])
	label_list = []
	for i in range(len(img_list)):
		if (i + 1) % 500 == 0:
			print("processing {} / {}".format(i + 1, len(img_list)))
		img_file = img_dir + img_list[indice_mask[i]][0]
		img_class = img_list[indice_mask[i]][1]
		img = Image.open(img_file)
		#temp_list = [img]
		new_img_list[0].append(np.asarray(img))
		for i, size in enumerate(resize_list):
			resized_img = img.resize(size)
			new_img_list[i + 1].append(np.asarray(resized_img))
		label_list.append(img_class)
		#new_img_list.append(temp_list)
	return new_img_list, label_list
