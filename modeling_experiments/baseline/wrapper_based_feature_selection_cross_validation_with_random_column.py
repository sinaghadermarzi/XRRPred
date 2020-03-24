# reads the result of evaluation of features and based on thresholds select a set of features
import csv
from sklearn import linear_model as lm
from sklearn.preprocessing import MinMaxScaler
import numpy
import scipy
from sklearn.metrics import mean_squared_error,mean_absolute_error
from pandas import *
#from libraries.settings import *
from scipy.stats.stats import pearsonr
from matplotlib import pyplot
import itertools
import os
import random
import math
from sklearn.svm import SVR


from sklearn import preprocessing



prediction_results = DataFrame()
df = pandas.read_csv('../datasets/unsampled/r-free/train_set_rfree.txt',delimiter = '\t')
test_set_file_address = '../datasets/unsampled/r-free/test_set_rfree.txt'
tdf = pandas.read_csv(test_set_file_address, delimiter="\t")
Y_test = numpy.transpose(numpy.array([tdf.loc[:, 'OUTPUT_R.Free']]))
Y_train = numpy.transpose(numpy.array([df.loc[:, 'OUTPUT_R.Free']]))


###############################
###############################
##########the label columns

column  = []
column.append('TEST SET RESULTS')
column.append('MAE')
column.append("MSE")
column.append("Pearson Correlation")
column.append("Spearman Correlation")
column.append('TRAINING_SET_RESULTS')
column.append('MAE')
column.append("MSE")
column.append("Pearson Correlation")
column.append("Spearman Correlation")
column.append('list of features')
column.append('number of features')
column.append('PREDICTIONS ON TEST SET')

test_IDs = numpy.transpose(numpy.array([tdf.loc[:, 'PDB ID']]))
for i in test_IDs:
    column.append(i[0])

prediction_results["Labels"] = Series(column)







column  = []
column.append('')
column.append('')
column.append("")
column.append("")
column.append("")
column.append('')
column.append('')
column.append("")
column.append("")
column.append("")
column.append('')
column.append('')
column.append('')

Native_res = numpy.transpose(numpy.array([tdf.loc[:, 'OUTPUT_R.Free']]))
for i in Native_res:
    column.append(i[0])

prediction_results['OUTPUT_R.Free'] = Series(column)







##############################################################
##############################################################
##############################################################
#######Random predictor with same-dist (shuffle)

column  =[]
import copy
import random
Y_pred_rand_same_dist = [None]*len(Y_test)

# numpy.random.shuffle(Y_pred_rand_same_dist)

for i in range(len(Y_test)):
    idx = numpy.random.randint(0,len(Y_train))
    Y_pred_rand_same_dist[i] = Y_train[idx]

Y_test.reshape(-1,1)
Y_pred_rand_same_dist = numpy.array(Y_pred_rand_same_dist)
Y_pred_rand_same_dist.reshape(-1,1)
for i in Y_test:
    if math.isnan(i) or i>100000:
        a = 7
MAE = mean_absolute_error(Y_pred_rand_same_dist,Y_test)
MSE = mean_squared_error(Y_pred_rand_same_dist,Y_test)
rho, p_val = scipy.stats.spearmanr(Y_pred_rand_same_dist,Y_test)
r,p_val = scipy.stats.pearsonr(Y_pred_rand_same_dist,Y_test)


column.append('TEST SET RESULTS')
column.append(str(MAE))
column.append(str(MSE))
column.append(str(r[0]))
column.append(str(rho))
column.append('TRAINING_SET_RESULTS')
column.append('')
column.append('')
column.append('')
column.append('')
column.append('')
column.append('')
column.append('PREDICTIONS ON TEST SET')
for i in Y_pred_rand_same_dist:
    column.append(str(i[0]))

prediction_results["rand same-dist"] = Series(column)


prediction_results.to_csv('result_report_with_baseline.csv',index = False,quoting=csv.QUOTE_ALL)






