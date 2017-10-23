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
print('\n')  # print section break)

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

    inverse_covariance_matrix = get_inverse_covariance_matrix(training_input, k_clusters, label)

    design_matrix = compute_design_matrix(training_input, centroids.T, inverse_covariance_matrix)
    # print(design_matrix.shape)

    weights = closed_form_solution(1, design_matrix, training_target)
    # print('Closed form solution: ', weights)

    validation_design_matrix = compute_design_matrix(validation_input, centroids.T, inverse_covariance_matrix)

    """ Gradient Descent Section """
    weights = SGD_sol(1, weights, 500, 0.1, design_matrix, training_target, validation_design_matrix, validation_target)
    # print('Gradient descent weights: \n', weights)

    """ Gradient Descent Section  """

"""
:description: the function generates the variance of the clustered data
:returns variance matrix
"""
def get_inverse_covariance_matrix(training_input,k_clusters, label):
    cluster_points = [[]] * k_clusters
    cluster_variance = [[]] * k_clusters
    for i in range(0, k_clusters):
        cluster_points[i] = (training_input[np.where(label == i)])
        cluster_variance[i] = np.linalg.pinv(np.identity(46) * np.var(cluster_points[i], axis=0, ddof=1))

    return np.array(cluster_variance)


def compute_design_matrix(input_data, cluster_centers, spread):
    cluster_centers = np.array([cluster_centers]).T
    broadcast = np.broadcast(input_data.T, cluster_centers)
    yj = np.empty(broadcast.shape)
    yj.flat = [x - u for (x, u) in broadcast]

    design_matrix = np.exp(
        np.sum(
            (np.einsum('ndm,mdd->ndm', yj.T, spread) * yj.T), axis=1)
        / (-2)
    )

    design_matrix = np.insert(design_matrix, 0, 1, axis=1)
    return design_matrix

"""
:description the function calculates the weights using the closed form solution.
             the regularizer term is used to minimize the overfitting issue 
:parameter regularizer lambda, design matrix, target data
:returns the weights calculated
"""
def closed_form_solution(regularizer_lambda, design_matrix, target_data):
    first_term = np.dot(regularizer_lambda, np.identity(len(design_matrix[0])))
    second_term = np.matmul(design_matrix.T, design_matrix)
    third_term = np.matmul(design_matrix.T, target_data)

    weights = np.linalg.solve(first_term + second_term, third_term).flatten()

    e_rms = compute_sum_of_squared_error(design_matrix, target_data, regularizer_lambda, weights)
    print('Closed form Erms: ', e_rms)
    return weights


def compute_sum_of_squared_error(design_matrix, target_data, regularizer_lambda, weights):
    error_term = np.sum(np.square(target_data - np.matmul(design_matrix, weights))) / 2
    regularizer_term = (np.matmul(weights.T, weights) * regularizer_lambda) / 2

    sum_of_squares_error = error_term + regularizer_term

    e_rms = np.sqrt(2 * sum_of_squares_error / len(design_matrix))
    return e_rms


def compute_expected_output(weights, design_matrix):
    expected_output = np.matmul(weights, design_matrix)
    return expected_output


def compute_gradient_error(design_matrix, target_data, weights, L2_lambda, size):
    yj = np.matmul(design_matrix, weights.T)
    difference = (yj - target_data).T
    e_d = np.matmul(difference, design_matrix)
    differentiation_error = (e_d + L2_lambda * weights) / size
    return differentiation_error


def SGD_sol(learning_rate, weights, minibatch_size, L2_lambda, design_matrix, target_data, validation_design_matrix, validation_target):
    N, _ = design_matrix.shape
    patience = 50
    improvement_threshold = 0.0001
    min_validation_error = np.inf
    optimal_weights = weights.shape
    j = 0

    while j < patience:
        for i in range(minibatch_size):
            lower_bound = i * minibatch_size
            upper_bound = min((i + 1) * minibatch_size, N)
            phi = design_matrix[lower_bound:upper_bound, :]
            t = target_data[lower_bound: upper_bound]
            differentiation_error = compute_gradient_error(phi, t, weights, L2_lambda, minibatch_size)
            weights = weights - learning_rate * differentiation_error

        validation_error = compute_gradient_error(validation_design_matrix, validation_target, weights, L2_lambda, len(validation_target))
        validation_error = np.linalg.norm(validation_error)
        if np.absolute(validation_error - min_validation_error) < improvement_threshold:
            min_validation_error = validation_error
            optimal_weights = weights
            break

        if validation_error < min_validation_error:
            j = 0
            min_validation_error = validation_error
            optimal_weights = weights
        else:
            j = j + 1

    erms = compute_sum_of_squared_error(design_matrix, target_data, L2_lambda, optimal_weights)
    print('SGD Erms: ', erms)

    return optimal_weights

if __name__ == '__main__':
    main()