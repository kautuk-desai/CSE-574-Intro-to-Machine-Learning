from PIL import Image
import numpy as np

class Utilities(object):
	"""docstring for ClassName"""
	def __init__(self, file_path):
		super(Utilities, self).__init__()
		self.file_path = file_path
		self.image_size = (28,28)

	def load_images(self, num_images, file_names):
		data = [[]]*num_images
		for i in range(num_images):
			im = Image.open(self.file_path + file_names[i])
			im = im.convert(mode='L')
			resized_im = im.resize(self.image_size)
			# resized_im.show()
			flattened_im = np.asarray(resized_im).flatten()
			data[i] = flattened_im

		return data
		