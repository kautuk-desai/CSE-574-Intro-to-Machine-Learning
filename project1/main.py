import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# print the name and person number
print('UBitName = ', 'satyasiv')
print('personNumber = ', 50248987)
print('UBitName = ', 'kautukra')
print('personNumber = ', 50247648)
print('\n')  # print section break


# variables related to file information
excel_sheetname = 'university_data'
excel_filename = 'university data.xlsx'
num_of_variables = 4

# get the relative file path as the file is in data set folder
current_dir = os.path.dirname(__file__)
file_path = os.path.abspath(os.path.join(current_dir, './DataSet/' + excel_filename))

# read the file using pandas and create a dataframe
dataframe = pd.DataFrame()
dataset = pd.read_excel(file_path, excel_sheetname)
dataframe = dataframe.append(dataset)
dataframe = dataframe[dataframe["CS Score (USNews)"].notnull()]

# get the list of variables from dataframe
cs_score_list = dataframe.ix[:, 2]  # CS Score (USNews) column index in dataframe
research_overhead_list = dataframe.ix[:, 3]  # Research Overhead % column index in dataframe
admn_base_pay_list = dataframe.ix[:, 4]  # Admin Base Pay$ column index
tuition_list = dataframe.ix[:, 5]  # Tuition(out-state)$ column index


# calculate mean for each variables
mu1 = round(np.mean(cs_score_list),3)
mu2 = round(np.mean(research_overhead_list),3)
mu3 = round(np.mean(admn_base_pay_list),3)
mu4 = round(np.mean(tuition_list),3)

# calculate variance for each variables
var1 = round(np.var(cs_score_list, ddof=1), 3)
var2 = round(np.var(research_overhead_list, ddof=1), 3)
var3 = round(np.var(admn_base_pay_list, ddof=1), 3)
var4 = round(np.var(tuition_list, ddof=1), 3)

# calculate standard deviation for each variables
sigma1 = round(np.std(cs_score_list, ddof=1), 3)
sigma2 = round( np.std(research_overhead_list, ddof=1), 3)
sigma3 = round(np.std(admn_base_pay_list, ddof=1), 3)
sigma4 = round(np.std(tuition_list, ddof=1), 3)

# print the mean, variance and standard deviation of each variables
print('mu1 = ', mu1)
print('mu2 = ', mu2)
print('mu3 = ', mu3)
print('mu4 = ', mu4)

print('var1 = ', var1)
print('var2 = ', var2)
print('var3 = ', var3)
print('var4 = ', var4)

print('sigma1 = ', sigma1)
print('sigma2 = ', sigma2)
print('sigma3 = ', sigma3)
print('sigma4 = ', sigma4)

# calculating covariance and correlation matrices
df_list = list();
df_list.append(cs_score_list)
df_list.append(research_overhead_list)
df_list.append(admn_base_pay_list)
df_list.append(tuition_list)
df_list_len = len(df_list)


# could have initialized in the same line i.e  covarianceMat = correlationMat = np.zeros([df_list_len, df_list_len]).
# but python lists and dictionaries are mutable objects
covarianceMat = np.zeros([df_list_len, df_list_len])
correlationMat = np.zeros([df_list_len, df_list_len])

covarianceMat = np.round(np.cov(df_list, ddof=1), 3)
correlationMat = np.round(np.corrcoef(df_list), 3)

# incase you want to use pandas cov function to calculate covariance matrice
# for i in range(df_list_len):
#     for j in range(i, df_list_len):
#
#         # calculate covariance
#         covarianceMat[i, j] = df_list[i].cov(df_list[j])
#         covarianceMat[j, i] = covarianceMat[i, j]
#
#         # calculate correlation
#         correlationMat[i, j] = df_list[i].corr(df_list[j])
#         correlationMat[j, i] = correlationMat[i, j]

print('covarianceMat = \n', covarianceMat)
print('correlationMat = \n', correlationMat)

# Calculate log likelihood of the given dataset
vector_of_points = np.zeros(num_of_variables)
logLikelihood = 0
logLikelihood_depenedent_var = 0
logpdf_independent_var = 0
logpdf_dependent_var = 0

covar_independent_var = covarianceMat * np.identity(num_of_variables)
vector_of_mean = [mu1, mu2, mu3, mu4]

for i in range(len(cs_score_list)):
    for j in range(df_list_len):
        vector_of_points[j] = (df_list[j][i])
    logpdf_independent_var = multivariate_normal.logpdf(vector_of_points, vector_of_mean,
                                                        covar_independent_var, allow_singular=True)
    logpdf_dependent_var = multivariate_normal.logpdf(vector_of_points, vector_of_mean,
                                                      covarianceMat, allow_singular=True)
    # if not(math.isnan(logpdf_independent_var)):
    logLikelihood += logpdf_independent_var
    logLikelihood_depenedent_var += logpdf_dependent_var
    vector_of_points = np.zeros(num_of_variables)

print('logLikelihood =', logLikelihood)
print('logLikelihood (Dependent variables) =', logLikelihood_depenedent_var)

# plot the
# colors = np.random.rand(49)
# f, axs = plt.subplots(3, 2,  figsize=(100, 100))
# i = 0
# j = 1
# for row in axs:
#     for col in row:
#
#         col.scatter(df_list[i], df_list[j], c=colors)
#         col.set_xlabel(df_list[i].name, fontsize=14)
#         col.set_ylabel(df_list[j].name, fontsize=14)
#         for k in range(50):
#             col.annotate(dataframe["name"][k], xy=(df_list[i][k],df_list[j][k]+0.7))
#         if j + 1 < 4:
#             j = j + 1
#         else:
#             j = i + 1
#     i = i + 1
#
# plt.show()

colors = np.random.rand(49)
subplot_index = 1
for i in range(4):
    for j in range(i + 1,4):
        f,axs = plt.subplots(1,1, figsize=(15,15))
        plt.scatter(df_list[i],df_list[j], c=colors)
        plt.xlabel(df_list[i].name, fontsize=14)
        plt.ylabel(df_list[j].name, fontsize=14)
        subplot_index+=1

plt.show()