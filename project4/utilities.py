class Utilities(object):
	"""docstring for ClassName"""
	def __init__(self, file_path):
		super(Utilities, self).__init__()
		self.file_path = file_path

	def load_images(self, num_images):
		data = [[]]*num_images
		for i in range(training_count):
			im = Image.open(self.file_path + celeba_train_img_file_names[i])
			im = im.convert(mode='L')
			resized_im = im.resize(image_size)
			# resized_im.show()
			flattened_im = np.asarray(resized_im).flatten()
			data[i] = flattened_im

		return data
		