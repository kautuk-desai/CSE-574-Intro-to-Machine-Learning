# import re
# import os

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2

# print the UBIT name and person number
print('UBitName = ', 'satyasiv')
print('personNumber = ', 50248987)
print('UBitName = ', 'kautukra')
print('personNumber = ', 50247648)
print('\n')  # print section break

"""
This code was written to clean the data in Querylevelnorm.txt. Now since we have the input data variables X and
target variable T in the file Querylevelnorm_X and Querylevelnorm_t respectively. We do not use the below code.
 
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

letor_dataset = np.genfromtxt('cleaned_dataset.txt', dtype=None, delimiter=" ")

End """


def main():

    # input data file does not contain the reference label which is our target. also it does not have the qid (column 2)
    # basically the file contains the cleaned data. It contains the feature values
    letor_input = np.genfromtxt('Querylevelnorm_X.csv', delimiter=',')

    # target datafile contains the reference label which is our target value and which determines the relationship
    # between query and document. Higher the value better the relation.
    letor_target = np.genfromtxt('Querylevelnorm_t.csv', delimiter=',')

    # print(letor_input[:, 0])  # first column

    "Partition Data Section"
    data_count = len(letor_input)
    training_data_row_limit = math.floor(0.8 * data_count)
    validation_data_row_limit = training_data_row_limit + math.floor(0.1 * data_count)

    training_input = letor_input[:training_data_row_limit]
    training_target = letor_target[:training_data_row_limit]

    validation_input = letor_input[training_data_row_limit: validation_data_row_limit]
    validation_target = letor_target[training_data_row_limit: validation_data_row_limit]

    test_input = letor_input[validation_data_row_limit: data_count]
    test_target = letor_target[validation_data_row_limit: data_count]
    "Partition Data Section End"

    " Choose spread section "
    # plt.hist(training_input[:, 1])
    # plt.show()
    # plt.hist(training_input[:, 2])
    # plt.show()

    # To choose the spread for each of the basis functions the histogram of few features was taken.
    # We see that most of the points have frequency peaks between 0 and 0.2.
    # So we choose 0.1 as the spread for each of the basis functions.

    " Choose spread section End "

    k_clusters = 30
    centroids, label = kmeans2(training_input, k_clusters, minit='points')

    # genearate spread of each data point
    covariance_matrix = np.cov(training_input, rowvar=False)
    identity_matrix = np.identity(46)

    covariance_matrix_diag = covariance_matrix * identity_matrix

    basis_funct = compute_design_matrix(training_input, centroids.T, covariance_matrix_diag)
    print(covariance_matrix_diag)


def compute_design_matrix(input_data, cluster_centers, spread):

    input_data = np.reshape(input_data, [len(input_data), len(input_data[0])])
    cluster_centers = np.reshape(cluster_centers, [len(cluster_centers), len(cluster_centers[0])])
    print(cluster_centers[0])
    print(cluster_centers[1])
    broadcast = np.broadcast(np.array(input_data), np.array(cluster_centers))

    out = np.empty(broadcast.shape)
    out.flat = [x-u for (x,u) in broadcast]

    basis_func = np.exp(
        np.sum(
            np.matmul(input_data - cluster_centers, spread) * (input_data - cluster_centers), axis=2
        ) / (-2)
    ).T

    return np.insert(basis_func, 0, 1, axis=1)


if __name__ == '__main__':
    main()
