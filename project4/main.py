import numpy as np
import random
from PIL import Image
from groupmembers import print_group_members
from utilities import Utilities

def main():
	print_group_members()

	feature_for_classification = 'Eyeglasses'
	image_size = (28,28)
	data_file_path = './data/img_align_celeba/'

	## read label file
	label_file_name = './data/list_attr_celeba.txt'
	features = np.genfromtxt(label_file_name, skip_header=1, max_rows=1, dtype='str')
	print(len(features))
	# bad method but couldn't find any alternative
	feature_col_index = np.where(features == feature_for_classification)[0][0]
	print(feature_col_index)
	
	label = np.genfromtxt(label_file_name, dtype= 'str',skip_header=2, usecols=(0, feature_col_index))
	dataset_count = len(label)
	training_count = int(0.8 * dataset_count)
	test_count = dataset_count - int(0.2*dataset_count)
	# python's random module isn’t made to deal with numpy arrays
	# since it’s not exactly the same as nested python lists
	np.random.shuffle(label)
	image_file_names = label[:, 0]
	predicted_output = label[:, 1]
	# print(image_file_names)

	celeba_train_img_file_names = image_file_names[0:training_count]
	celeba_test_img_file_names = image_file_names[training_count:dataset_count]
	# print('train image count: ', len(celeba_train_img_file_names))
	# print('test image count: ', len(celeba_test_img_file_names))

	celeba_train_labels = predicted_output[0:training_count]
	celeba_test_labels = predicted_output[training_count:dataset_count]
	print('train labels: ', len(celeba_train_labels))
	print('test labels: ', len(celeba_test_labels))

	utilities = Utilities(data_file_path)
	celeba_train_images = utilities.load_images(training_count)
	celeba_test_images = utilities.load_images(test_count)

	print(celeba_train_images[0])
	

if __name__ == '__main__':
	main()