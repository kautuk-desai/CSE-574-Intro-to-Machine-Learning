import numpy as np
from groupmembers import print_group_members

def main():
	print_group_members()

	feature_for_classification = 'Eyeglasses'

	## read label file
	label_file_name = './data/list_attr_celeba.txt'
	features = np.genfromtxt(label_file_name, skip_header=1, max_rows=1, dtype='str')
	print(len(features))
	# bad method but couldn't find any alternative
	feature_col_index = np.where(features == feature_for_classification)[0][0]
	print(feature_col_index)
	
	label = np.genfromtxt(label_file_name, dtype= 'str',skip_header=2)
	# print(len(label[0]))


if __name__ == '__main__':
	main()