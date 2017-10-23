# import re
# import os

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2


def main():
    # compute_linear_reg_letor()

    compute_linear_reg_synthetic_data()


def compute_linear_reg_letor():
    # input data file does not contain the reference label which is our target. also it does not have the qid (column 2)
    # basically the file contains the cleaned data. It contains the feature values
    letor_input = np.genfromtxt('Querylevelnorm_X.csv', delimiter=',')
    global num_of_observations, num_of_features
    num_of_observations, num_of_features = letor_input.shape
    # target datafile contains the reference label which is our target value and which determines the relationship
    # between query and document. Higher the value better the relation.
    letor_target = np.genfromtxt('Querylevelnorm_t.csv', delimiter=',')

    partition_data(letor_input, letor_target)

    " Choose spread section "
    # plt.hist(training_input[:, 1])
    # plt.show()
    # plt.hist(training_input[:, 2])
    # plt.show()

    # To choose the spread for each of the basis functions the histogram of few features was taken.
    # We see that most of the points have frequency peaks between 0 and 0.2.
    # So we choose 0.1 as the spread for each of the basis functions.
    " Choose spread section End "

    # compute_optimal_hyperparameters()

    # the hyperparameters are obtained after computing the search on range of M (1 to 100) and lambda (0 to 1, 0.1)
    model_complexity = 56
    regularizer_lambda = 0

    centroids, label = kmeans2(training_input, model_complexity, minit='points')
    inv_covariance_matrix = get_inverse_covariance_matrix(model_complexity, label)
    design_matrix = compute_design_matrix(training_input, centroids.T, inv_covariance_matrix)
    closed_form_weights = closed_form_solution(design_matrix, training_target, regularizer_lambda)

    # Gradient Descent Section
    validation_design_matrix = compute_design_matrix(validation_input, centroids.T, inv_covariance_matrix)
    sgd_weights = compute_SGD(design_matrix, training_target, validation_design_matrix, validation_target, regularizer_lambda)
    # Gradient Descent Section

    compute_test_result(centroids, inv_covariance_matrix, closed_form_weights, sgd_weights, regularizer_lambda)


def compute_linear_reg_synthetic_data():
    synthetic_input = np.genfromtxt('input.csv', delimiter=',')
    global num_of_observations, num_of_features
    num_of_observations, num_of_features = synthetic_input.shape

    synthetic_target = np.genfromtxt('output.csv', delimiter=',')

    partition_data(synthetic_input, synthetic_target)

    compute_optimal_hyperparameters()

    model_complexity = 30
    regularizer_lambda = 0

    centroids, label = kmeans2(training_input, model_complexity, minit='points')
    inv_covariance_matrix = get_inverse_covariance_matrix(model_complexity, label)
    design_matrix = compute_design_matrix(training_input, centroids.T, inv_covariance_matrix)
    closed_form_weights = closed_form_solution(design_matrix, training_target, regularizer_lambda)

    # Gradient Descent Section
    validation_design_matrix = compute_design_matrix(validation_input, centroids.T, inv_covariance_matrix)
    sgd_weights = compute_SGD(design_matrix, training_target, validation_design_matrix, validation_target,
                              regularizer_lambda)
    # Gradient Descent Section


    compute_test_result(centroids, inv_covariance_matrix, closed_form_weights, sgd_weights, regularizer_lambda)



def partition_data(input_data, target_data):
    training_data_row_limit = math.floor(0.8 * num_of_observations)
    validation_data_row_limit = training_data_row_limit + math.floor(0.1 * num_of_observations)

    global training_input, training_target, validation_input, validation_target, test_input, test_target
    training_input = input_data[:training_data_row_limit]
    training_target = target_data[:training_data_row_limit]

    validation_input = input_data[training_data_row_limit: validation_data_row_limit]
    validation_target = target_data[training_data_row_limit: validation_data_row_limit]

    test_input = input_data[validation_data_row_limit: num_of_observations]
    test_target = target_data[validation_data_row_limit: num_of_observations]

"""
:returns the optimal model complexity (M) for Gaussian RBF
"""
def compute_optimal_hyperparameters():
    k_clusters = 10
    min_lambda = 0
    max_lambda = 1
    lambda_step_size = 0.1
    optimal_M = np.inf
    optimal_lambda = np.inf
    optimal_centroids = None
    optimal_error_rms = np.inf
    optimal_weights = None
    error_rms_grid = np.zeros([k_clusters, int(max_lambda / lambda_step_size)])
    test_lambda = np.arange(min_lambda, max_lambda, lambda_step_size)

    for k_cluster in range(1, k_clusters):
        centroids, label = kmeans2(training_input, k_cluster, minit='points')
        inv_covariance_matrix = get_inverse_covariance_matrix(k_cluster, label)
        design_matrix = compute_design_matrix(training_input, centroids.T, inv_covariance_matrix)
        validation_design_matrix = compute_design_matrix(validation_input, centroids.T, inv_covariance_matrix)

        for index, _lambda in enumerate(test_lambda):
            closed_form_weights = closed_form_solution(design_matrix, training_target, _lambda)
            error_rms = compute_sum_of_squared_error(validation_design_matrix, validation_target, closed_form_weights, _lambda)
            error_rms_grid[k_cluster, index] = error_rms
            print('Error for M = {0} and lambda = {1} is {2}'.format(k_cluster, _lambda, error_rms))
            if error_rms < optimal_error_rms:
                optimal_error_rms = error_rms
                optimal_M = k_cluster
                optimal_lambda = _lambda

    print('Optimal model complexity (M) for M in 1 to {} is: {}'.format(k_clusters, optimal_M))
    print('optimal regularizer constant lambda in {} to {} is: {}'.format(min_lambda, max_lambda, optimal_lambda))



"""
:description: the function generates the variance of the clustered data
:returns variance matrix
"""
def get_inverse_covariance_matrix(k_clusters, label):
    cluster_points = [[]] * k_clusters
    cluster_variance = [[]] * k_clusters
    for i in range(0, k_clusters):
        cluster_points[i] = (training_input[np.where(label == i)])
        cluster_variance[i] = np.linalg.pinv(np.identity(num_of_features) * np.var(cluster_points[i], axis=0, ddof=1))

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
def closed_form_solution(design_matrix, target_data, regularizer_lambda):
    first_term = np.dot(regularizer_lambda, np.identity(len(design_matrix[0])))
    second_term = np.matmul(design_matrix.T, design_matrix)
    third_term = np.matmul(design_matrix.T, target_data)
    weights = np.linalg.solve(first_term + second_term, third_term).flatten()
    return weights


def compute_sum_of_squared_error(design_matrix, target_data, weights, regularizer_lambda):
    error_term = np.sum(np.square(target_data - np.matmul(design_matrix, weights))) / 2
    regularized_term = (np.matmul(weights.T, weights) * regularizer_lambda) / 2
    sum_of_squares_error = error_term + regularized_term
    e_rms = np.sqrt(2 * sum_of_squares_error / len(design_matrix))
    return e_rms


def compute_expected_output(weights, design_matrix):
    expected_output = np.matmul(weights, design_matrix)
    return expected_output


def compute_gradient_error(design_matrix, target_data, weights, regularizer_lambda):
    yj = np.matmul(design_matrix, weights.T)
    difference = (yj - target_data).T
    e_d = np.matmul(difference, design_matrix)
    differentiation_error = (e_d + regularizer_lambda * weights) / mini_batch_size
    return differentiation_error


def compute_SGD(design_matrix, target_data, validation_design_matrix, validation_target, regularizer_lambda):
    N, _ = design_matrix.shape
    patience = 50  # this is our patience level!
    min_validation_error = np.inf
    weights = np.zeros(design_matrix.shape[1])
    optimal_weights = np.zeros(design_matrix.shape[1])
    j = 0
    steps = int(N / mini_batch_size)

    for epoch in range(num_epochs):
        while j < patience:
            for i in range(steps):
                lower_bound = i * mini_batch_size
                upper_bound = min((i + 1) * mini_batch_size, N)
                phi = design_matrix[lower_bound:upper_bound, :]
                t = target_data[lower_bound: upper_bound]

                differentiation_error = compute_gradient_error(phi, t, weights, regularizer_lambda)
                weights = weights - learning_rate * differentiation_error

            validation_error_rms = compute_sum_of_squared_error(validation_design_matrix, validation_target, weights,
                                                                regularizer_lambda)

            if validation_error_rms < min_validation_error:
                j = 0
                min_validation_error = validation_error_rms
                optimal_weights = weights
            else:
                j = j + 1

    return optimal_weights


def compute_test_result(centroids, inverse_covariance_matrix, closed_form_weights, sgd_weights, regularizer_lambda):

    test_design_matrix = compute_design_matrix(test_input, centroids.T, inverse_covariance_matrix)
    test_error_rms_cf = compute_sum_of_squared_error(test_design_matrix, test_target, closed_form_weights,
                                                     regularizer_lambda)
    print('Test Error rms (closed form weights): ', test_error_rms_cf)

    test_error_rms_sgd = compute_sum_of_squared_error(test_design_matrix, test_target, sgd_weights, regularizer_lambda)
    print('Test Error rms (stochastic gradient decent): ', test_error_rms_sgd)

    y_closed_form = np.matmul(test_design_matrix, closed_form_weights)
    y_sgd = np.matmul(test_design_matrix, sgd_weights)
    # print(y_closed_form)

def compute_validation_result():
    print('test')

if __name__ == '__main__':
    # initialize common variables that are going to be used through-out
    learning_rate = 1
    num_epochs = 100
    mini_batch_size = 500
    num_of_observations, num_of_features = 0, 0
    training_input = 0
    training_target = 0
    validation_input = None
    validation_target = None
    test_input = None
    test_target = None
    main()
