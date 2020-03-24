
import pandas
import itertools
import numpy
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics.regression import mean_absolute_error,mean_squared_error

from sklearn.preprocessing import StandardScaler





def MAE(Y1,Y2):
    return mean_absolute_error(Y1,Y2)
def MSE(Y1, Y2):
    return mean_squared_error(Y1, Y2)
def SPC(y_true, y_pred):
    corr , _ = spearmanr(y_true,y_pred)
    return corr
def PNC(y_true, y_pred):
    corr , _ = pearsonr(y_true,y_pred)
    return corr




def normalize_df(df,features,target_col,scaler):
    X = df.loc[:,features].values
    Y = df.loc[:,target_col].values
    X = scaler.transform(X)
    out_df = pandas.DataFrame(X,index = df.index, columns=features)
    out_df[target_col]= Y
    return out_df



def normalize_train_test(training_set_df , test_set_df, feature_cols, target_col):
    train_set_X = training_set_df.loc[ :, feature_cols].values
    scaler = StandardScaler()
    scaler.fit(train_set_X)
    test_set_norm_df = normalize_df(test_set_df,input_features,target_col,scaler)
    training_set_norm_df = normalize_df(training_set_df,input_features,target_col,scaler)
    return training_set_norm_df, test_set_norm_df




def cross_validated_predictions(model,trainfolds_dfs,testfolds_dfs,feature_set,target_col_name):
    y = []

    y_pred  = []
    for i in range(len(testfolds_dfs)):
        train_X = trainfolds_dfs[i].loc[:,feature_set].values
        train_Y = trainfolds_dfs[i].loc[:,target_col_name].values
        test_X = testfolds_dfs[i].loc[:,feature_set].values
        test_Y = testfolds_dfs[i].loc[:,target_col_name].values
        model.fit(train_X,train_Y)
        test_pred = model.predict(test_X)
        y += list(test_Y)
        y_pred += list(test_pred)
    return y,y_pred


def get_results_column(y,y_pred):
    #for the next step, I want to turn this function into a class.
    # something like evaluation_which you just set the target column and then add the result columns one by one
    # you can add arbitrary number of evaluation metrics and it will be able to output the end table at the end
    mae = mean_absolute_error(y,y_pred)
    mse = mean_squared_error(y,y_pred)
    sp = SPC(y,y_pred)
    pn = PNC(y,y_pred)
    m = numpy.mean(y_pred)
    s = numpy.std(y_pred)
    return [mae,mse,pn,sp,"",m,s,""]+list(y_pred)



def unpack_grid(grid_dict):
#each key is a parameter name and each value is the list of values that that parameter should take
    return [dict(zip(grid_dict.keys(),case)) for case in itertools.product(*grid_dict.values())]
#


# def optimize_params(train_fold_dfs,test_fold_dfs,input_features,target_col,eval_func,tolerance,maximize,param_vals):
#     current_best_params  = None
#     current_best_score = float("inf")
#     if maximize:
#         current_best_score = -current_best_score
#     for param_val in itertools.product(param_vals[0],param_vals[1],param_vals[2]):
#         print("param0= ",param_val[0])
#         print("param1= ", param_val[1])
#         print("param2= ", param_val[2])
#         print("\n")
#         # model = MLPRegressor(hidden_layer_sizes= (10,8,4,1), activation ="relu", alpha =param_vals[0],
#         #                      learning_rate= param_vals[1],momentum = param_vals[2])
#         model = MLPRegressor(hidden_layer_sizes= (10,8,4,1), alpha =param_val[0], learning_rate= param_val[1],momentum = param_val[2])
#         y,y_pred = cross_validated_predictions(model,train_fold_dfs,test_fold_dfs,input_features,target_col)
#         new_score = eval_func(y,y_pred)
#         print("score:",new_score,"\n\n")
#         if maximize:
#             if new_score - current_best_score >= tolerance:
#                 current_best_score = new_score
#                 current_best_params = param_val
#         else:
#             if current_best_score - new_score >= tolerance:
#                 current_best_score = new_score
#                 current_best_params = param_val
#
#     return current_best_params

import progressbar
def gridsearch_detailed_result(model,train_fold_dfs,test_fold_dfs,input_features,target_col,grid):
    model_name = type(model).__name__
    print(model_name,":")
    results_df = pandas.DataFrame()
    param_settings = unpack_grid(grid)
    num_settings = len(param_settings)
    # widgets = ['Test: ', progressbar.Percentage(), ' ', progressbar.Bar(marker='0', left='[', right=']'),
    #            ' ', progressbar.ETA(), ' ']
    # pb = progressbar.ProgressBar(maxval=num_settings, widgets=widgets)
    # pb.start()
    current_num = 0
    for param_setting in param_settings:
        #initialize a model with specified parameters
        # model.__init__()
        model.set_params(**param_setting)
        y,y_pred = cross_validated_predictions(model,train_fold_dfs,test_fold_dfs,input_features,target_col)
        param_vals_str = [str(val) for val in param_setting.values()]
        col_name = model_name+"_"+"_".join(param_vals_str )
        results_df[col_name] = get_results_column(y,y_pred)
        results_df[target_var] = ["", "", "", "", "","","",""] + list(y)
        current_num+=1
        # pb.update(current_num)
    print("\n")
    return results_df



def get_results_column_calc(model,train_fold_dfs,test_fold_dfs,input_features,target_col,params):
    model.set_params(**params)
    model_name = type(model).__name__
    y, y_pred = cross_validated_predictions(model, train_fold_dfs, test_fold_dfs, input_features, target_col)
    param_vals_str = [str(val) for val in params.values()]
    col_name = model_name + "_" + "_".join(param_vals_str)
    res = get_results_column(y, y_pred)
    return res,col_name

import multiprocessing
from joblib import Parallel, delayed
def gridsearch_detailed_result_parallel(model,train_fold_dfs,test_fold_dfs,input_features,target_col,grid):
    num_cores = multiprocessing.cpu_count()
    param_settings = unpack_grid(grid)
    # num_settings = len(param_settings)
    results_column_list = Parallel(n_jobs=num_cores)(delayed(get_results_column_calc)(model,train_fold_dfs,test_fold_dfs,input_features,target_col,params) for params in
                                                                                      param_settings)
    return results_column_list



def column_list_to_dataframe(column_list):
    df = pandas.DataFrame()
    for col,col_name in column_list:
        df[col_name] = col
    return df



###############################################

###### models and their grid
from sklearn.tree import DecisionTreeRegressor
# from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR

model_and_grid = [
    # (
    #     DecisionTreeRegressor,
    #     {
    #         "max_depth": [5, 10, 50, 100],
    #         "min_samples_split": [20, 50, 200]
    #     }
    # ),
    # (
    #     KernelRidge,
    #     {
    #         # "gamma": [1,2,4],
    #         "kernel": ["rbf","poly"]
    #     }
    # ),
    # (
    #     SVR,
    #     {
    #         "gamma" : [0.25,0.5,0.75,1],
    #         "C" : [0.25,0.5,1,5,10,100],
    #         "kernel" : ["rbf","poly"]
    #     }
    # ),

    (
        Lasso,
        {
            "alpha": [0.0001,0.001,0.01,0.1],
        }
    ),
    (
        ElasticNet,
        {
            "alpha": [0.0001,0.001,0.01,0.1],
            "l1_ratio": [0.1,0.2,0.4,0.8]
        }
    ),
    (
        BayesianRidge,
        {
            "alpha_1": [
                # 1e-4,
                1e-6,
                # 1e-8
            ],
            # "alpha_2": [1e-4,1e-6,1e-8],
            # "lambda_1": [1e-4,1e-6,1e-8],
            # "lambda_2": [1e-4,1e-6,1e-8],
        }
    ),
    # (
    #     ARDRegression,
    #     {
    #         "alpha_1": [1e-4, 1e-6, 1e-8],
    #         "alpha_2": [1e-4, 1e-6, 1e-8],
    #         "lambda_1": [1e-4, 1e-6, 1e-8],
    #         "lambda_2": [1e-4, 1e-6, 1e-8],
    #     }
    # ),
    (
        SGDRegressor,
        {
            "loss": [
                # "squared_loss",
                # "huber",
                "epsilon_insensitive",
                "squared_epsilon_insensitive"
            ],
            "penalty": [
                "l2",
                "l1",
                "elasticnet"
            ],
            "alpha": [
                0.001,
                0.0001,
                0.00001
            ],
            "l1_ratio": [
                0.5,
                0.15,
                0.35,
                0.7,
                0.9
            ]
        }
    ),
    (
        PassiveAggressiveRegressor,
        {
            "C": [0.1,0.2,0.4,0.8,1,2,4],
            "epsilon": [0.05,0.1,0.2,0.4,0.8]
        }
    ),
    (
        TheilSenRegressor,
        {
            "max_subpopulation": [5,10, 100, 1e3, 1e4],
            # "n_jobs":[-1]
        }
    ),
    # (
    #     GaussianProcessRegressor,
    #     {
    #         "alpha":[1e-10]
    #     }
    # ),


]

############## read configurations

import json
with open("configurations.json") as jsonfile:
    conf = json.load(jsonfile)
dataset_root_dir = conf["dataset_root_dir"]
sampling_methods = conf["sampling_methods"]
sampling_methods = [x+"_lowdim" for x in sampling_methods]
test_set_path = dataset_root_dir + conf["file_locations"]["test_set"]
test_folds_relpath = conf["file_locations"]["test_folds"]
test_folds_path = [dataset_root_dir + relpath for relpath in test_folds_relpath]
input_features = conf["input_features"]





#####################################

test_set_df = pandas.read_csv(test_set_path)
num_CV_folds = len(test_folds_path)

for target_var in ["resolution"]:
    # print("-------------",target_var)
    ##################################### prepare the relevant dataframes for dataset and cv folds
    results_df = pandas.DataFrame()
    target_col = conf["target_col"][target_var]
    input_features = conf["input_features"]["selected"][target_var]
    sampling = "nosampling"
    training_set_path = dataset_root_dir+conf["file_locations"]["training_set"][sampling][target_var]
    train_folds_relpath  = conf["file_locations"]["training_folds"][sampling][target_var]
    train_folds_path = [dataset_root_dir + relpath for relpath in train_folds_relpath]
    test_fold_norm_df = [None] * num_CV_folds
    train_fold_norm_df = [None] * num_CV_folds
    ids = []
    natives = []
    for i in range(num_CV_folds):
        trainf_df = pandas.read_csv(train_folds_path[i])
        testf_df = pandas.read_csv(test_folds_path[i])
        ids += list(testf_df["PDB ID"].values)
        natives += list(testf_df[target_col].values)
        train_fold_norm_df[i],test_fold_norm_df[i] = normalize_train_test(trainf_df,testf_df, input_features,
                                                                          target_col)

    train_set_norm_df, test_set_norm_df = normalize_train_test(pandas.read_csv(training_set_path),
                                                                   pandas.read_csv(test_set_path), input_features,
                                                                   target_col)
    #initialize the results dataframe
    # lbls = ["MAE","MSE", "PEARSON","SPEARMAN","","MEAN","STDEV",""]+ids
    result_df = pandas.DataFrame()
    # results_df[target_col] = ["", "", "", "", "","","",""] + list(test_set_df[target_col].values)
    labels_col = (["MAE","MSE", "PEARSON","SPEARMAN","","MEAN","STDEV",""]+ids,"row label")
    target_var_values_col = (["", "", "", "", "","","",""] + natives,target_col)
    current_columns = [labels_col,target_var_values_col]#labels and native values
    #obtain detailed results for each model and its specified grid
    for modeltype,grid in model_and_grid:
        model = modeltype()
        model_name = type(model).__name__
        print(model_name)
        new_res = gridsearch_detailed_result_parallel(model,train_fold_norm_df,test_fold_norm_df,input_features,target_col,grid)
        new_res_df = column_list_to_dataframe(new_res)
        new_res_df.to_csv("results_" + target_col +"_"+model_name+ "_lowdim.csv")
        current_columns += list(new_res)


    results_df = column_list_to_dataframe(current_columns)
    results_df.to_csv("results_"+target_col+"_lowdim.csv")








