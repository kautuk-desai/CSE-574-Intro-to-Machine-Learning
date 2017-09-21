import os
import pandas as pd
import numpy as np

# print the name and person number
print('UBitName = ', 'kautukra')
print('personNumber = ', 50247648)
print('\n') # print section break


# variables related to file information
excel_sheetname = 'university_data'
excel_filename = 'university data.xlsx'

# get the relative file path as the file is in data set folder
current_dir = os.path.dirname(__file__)
file_path = os.path.abspath(os.path.join(current_dir, './DataSet/' + excel_filename))

# read the file using pandas and create a dataframe
dataframe = pd.DataFrame()
dataset = pd.read_excel(file_path, excel_sheetname)
dataframe = dataframe.append(dataset)

# get the list of variables from dataframe
cs_score_list = dataframe.ix[:, 2]  # CS Score (USNews) column index in dataframe
research_overhead_list = dataframe.ix[:, 3]  # Research Overhead % column index in dataframe
admn_base_pay_list = dataframe.ix[:, 4]  # Admin Base Pay$ column index
tuition_list = dataframe.ix[:, 5]  # Tuition(out-state)$ column index


# calculate mean for each variables
mu1 = np.mean(cs_score_list)
mu2 = np.mean(research_overhead_list)
mu3 = np.mean(admn_base_pay_list)
mu4 = np.mean(tuition_list);

# calculate variance for each variables
var1 = np.var(cs_score_list)
var2 = np.var(research_overhead_list)
var3 = np.var(admn_base_pay_list)
var4 = np.var(tuition_list)

# calculate standard deviation for each variables
sigma1 = np.std(cs_score_list)
sigma2 = np.std(research_overhead_list)
sigma3 = np.std(admn_base_pay_list)
sigma4 = np.std(tuition_list)

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

covarianceMat = np.zeros([df_list_len, df_list_len])
correlationMat = np.zeros([df_list_len, df_list_len])  # could have initialized in the same line i.e  covarianceMat = correlationMat = np.zeros([df_list_len, df_list_len]).
                                                       # but python lists and dictionaries are mutable objects
for i in range(df_list_len):
    for j in range(i, df_list_len):

        # calculate covariance
        covarianceMat[i, j] = df_list[i].cov(df_list[j])
        covarianceMat[j, i] = covarianceMat[i, j]

        # calculate correlation
        correlationMat[i, j] = df_list[i].corr(df_list[j])
        correlationMat[j, i] = correlationMat[i, j]

print('covarianceMat = \n', covarianceMat)
print('correlationMat = \n', correlationMat)




