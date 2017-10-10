import re
import pandas as pd
import numpy as np
import os

# print the UBIT name and person number
print('UBitName = ', 'satyasiv')
print('personNumber = ', 50248987)
print('UBitName = ', 'kautukra')
print('personNumber = ', 50247648)
print('\n')  # print section break

# variables realted to file
dataset_filename = 'Querylevelnorm.txt'

# get the relative file path as the file is in data set folder
current_dir = os.path.dirname(__file__)
file_path = os.path.abspath(os.path.join(current_dir, './MQ2007/' + dataset_filename))

# since each row in text file contains feature number and :
# the file is opened and unwated characters are removed from the file text
with open(file_path, 'r') as dataset_file:
    text = dataset_file.read()
    text = re.sub(r"\d+:", "", text)
dataset_file.close()

# the cleaned file text is then saved to a new file
with open('cleaned_dataset.txt', 'w') as file:
    file.write(text)
file.close()

dataset = np.genfromtxt('cleaned_dataset.txt', dtype=None, delimiter=" ")
