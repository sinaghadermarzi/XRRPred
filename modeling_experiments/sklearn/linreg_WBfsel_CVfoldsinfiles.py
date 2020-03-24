
import pandas
import sklearn
import itertools
import numpy
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics.regression import mean_absolute_error,mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from ast import literal_eval
# from read_config import read_conf





import pandas
import itertools
import numpy
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics.regression import mean_absolute_error,mean_squared_error
from sklearn.tree import DecisionTreeRegressor
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






def avg_cross_validation_score(model,scorer,trainfolds_dfs,testfolds_dfs,feature_set,target_col_name):
    scores = [None]*len(testfolds_dfs)
    for i in range(len(testfolds_dfs)):
        train_X = trainfolds_dfs[i].loc[:,feature_set].values
        train_Y = trainfolds_dfs[i].loc[:,target_col_name].values
        test_X = testfolds_dfs[i].loc[:,feature_set].values
        test_Y = testfolds_dfs[i].loc[:,target_col_name].values
        model.fit(train_X,train_Y)
        test_pred = model.predict(test_X)
        scores[i] = scorer(test_pred,test_Y)
    # if float("nan") in scores:
    #     print("None seen in scores")
    avg_score  = numpy.mean(scores)
    if numpy.isnan(avg_score):
        print("None seen in scores")
    # print("cross validation on ",feature_set)
    # print("scores: ",scores)
    return avg_score



def get_resutls_column(model,trainfolds_dfs,testfolds_dfs,train_set,test_set,feature_set,target_col_name):
    MSEs = [None]*len(testfolds_dfs)
    MAEs = [None]*len(testfolds_dfs)
    SPs = [None]*len(testfolds_dfs)
    PNs = [None]*len(testfolds_dfs)
    for i in range(len(testfolds_dfs)):
        train_X = trainfolds_dfs[i].loc[:,feature_set].values
        train_Y = trainfolds_dfs[i].loc[:,target_col_name].values
        test_X = testfolds_dfs[i].loc[:,feature_set].values
        test_Y = testfolds_dfs[i].loc[:,target_col_name].values
        model.fit(train_X,train_Y)
        test_pred = model.predict(test_X)
        MAEs[i] = MAE(test_pred,test_Y)
        MSEs[i] = MSE(test_pred, test_Y)
        SPs[i] = SPC(test_pred, test_Y)
        PNs[i] = PNC(test_pred, test_Y)

    train_cvavg_MAE  = numpy.mean(MAEs)
    train_cvavg_MSE = numpy.mean(MSEs)
    train_cvavg_PN = numpy.mean(PNs)
    train_cvavg_SP = numpy.mean(SPs)


    test_Y = test_set.loc[:, target_col].values
    test_X = test_set.loc[:, feature_set].values
    train_X = train_set.loc[:, feature_set].values
    train_Y = train_set.loc[:, target_col].values

    model.fit(train_X, train_Y)
    test_pred = model.predict(test_X)

    testset_pn, _ = pearsonr(test_Y, test_pred)
    testset_sp, _ = spearmanr(test_Y, test_pred)
    testset_mae = mean_absolute_error(test_Y, test_pred)
    testset_mse = mean_squared_error(test_Y, test_pred)

    column = [testset_mae, testset_mse,testset_pn, testset_sp, train_cvavg_MAE, train_cvavg_MSE, train_cvavg_PN, train_cvavg_SP ,feature_set,len(feature_set)]
    column += list(test_pred)
    return column


def wrapper_optimize(model,eval_criterion,input_sorted_features,train_fold_dfs,test_fold_dfs,target_column_name):
    # input of the wrapper-based method is
    #   a pairs evaluation method and its threshold of increase in that score
    #   a threshold for the correlation of features which pass the filter or in other words features to consider
    #   a threshold for the mutual correlation between features to be acceptable
    # output of the wrapper based method
    #   the list of best features
    #   it can also print the progress of the wrapper based
    eval_func, eval_th , maximize = eval_criterion
    current_best_score  = float("inf")
    current_features = []
    if maximize:
        current_best_score = - current_best_score
    for feature in input_sorted_features:
        feautres_including_new = current_features+[feature]
        #find the resutls of cross validation for that set of features
        new_score =  avg_cross_validation_score(model,globals()[eval_func],train_fold_dfs,test_fold_dfs,feautres_including_new,target_column_name)
        if maximize:
            if new_score - current_best_score >= eval_th:
                current_best_score = new_score
                current_features = feautres_including_new
        else:
            if current_best_score - new_score >= eval_th:
                current_best_score = new_score
                current_features = feautres_including_new
    return current_features, current_best_score




def filter_and_sort_features (dataset,input_features,target_col,filter_correlation_threshold,mutual_correlation_threshold):
    #this function is supposed to get a dataset (examples in rows and feautres in columns)
    #it outputs a list of noncorrelated (<mutual_correlation_threshold)
    #these are feautres with highest correlation and they have (absolute) correlation of greater than
    # filter_correlation_threshold
    scores  = []
    target_vals = dataset.loc[:,target_col].values
    filtered_features = []
    for f in input_features:
        feat_vals = dataset.loc[:,f].values
        corr,_ = spearmanr(target_vals,feat_vals)
        score = abs(corr)
        if score>filter_correlation_threshold:
            filtered_features.append(f)
            scores.append(score)
    scores  = numpy.array(scores)
    sorted_idx  = numpy.argsort(scores)[::-1]
    sorted_features = numpy.array(filtered_features)[sorted_idx]
    sorted_scores = scores[sorted_idx]

    accepted_features = []
    for f in sorted_features:
        redundant = False
        new_feat_vals = dataset.loc[:, f].values
        for ff in  accepted_features:
            accepted_feat_vals = dataset.loc[:, ff].values
            corr, _ = spearmanr(new_feat_vals,accepted_feat_vals)
            if abs(corr) > mutual_correlation_threshold:
                redundant = True
        if not redundant:
            accepted_features.append(f)

    return accepted_features










############## read configurations
import json
with open("configurations.json") as jsonfile:
    conf = json.load(jsonfile)
dataset_root_dir = conf["dataset_root_dir"]
sampling_methods = conf["sampling_methods"]
test_set_path = dataset_root_dir + conf["file_locations"]["test_set"]
test_folds_relpath = conf["file_locations"]["test_folds"]
test_folds_path = [dataset_root_dir + relpath for relpath in test_folds_relpath]
input_features = conf["input_features"]


wrapper_opt_step = {}
mutual_th_ops = {}
filt_thresholds ={}

wrapper_opt_step["resolution"] = [("SPC",0.01,True),
    ("SPC",0.0075,True),
    ("SPC",0.005,True),
    ("SPC",0.001,True),
    #("PNC",0.01,True),
    ("MAE",0.1,False),
    ("MAE",0.05, False),
    ("MAE",0.01,False),
    ("MAE",0.005, False),
    #("MSE",0.1,False),
    #("MWE",0.05,False),
    #("M4E",0.5,False)
    ]

mutual_th_ops["resolution"] = [0.5,0.7,1]
filt_thresholds["resolution"] = [0,0.1,0.2,0.3]





wrapper_opt_step["rfree"]  = [("SPC",0.01,True),
    ("SPC",0.0075,True),
    ("SPC",0.005,True),
    ("SPC",0.001,True),
    #("PNC",0.01,True),
    #("MAE",0.1,False),
    ("MAE",0.05, False),
    ("MAE",0.01, False),
    ("MAE",0.005, False),
	("MAE",0.001, False),
    #("MSE",0.1,False),
    #("MWE",0.05,False),
    #("M4E",0.5,False)
    ]
mutual_th_ops["rfree"] = [0.5,0.7,1]
filt_thresholds["rfree"] = [0,0.1,0.2]




#####################################


test_set_df = pandas.read_csv(test_set_path)
num_CV_folds = len(test_folds_path)



row_labels  = ["test_MAE","test_MSE","test_pearson","test_spearman","avg_training_MAE","avg_test_MSE","avg_training_pearson","avg_training_spearman","feature_list","num_features"]+list(test_set_df["PDB ID"].values)



for target_var in ["rfree"]:

    target_col = conf["target_col"][target_var]

    for sampling in sampling_methods:
        results_df = pandas.DataFrame()
        results_df[target_col] = ["", "", "", "", "", "", "", "", "", ""] + list(test_set_df[target_col].values)
        results_df.index = row_labels

        training_set_path = dataset_root_dir+conf["file_locations"]["training_set"][sampling][target_var]
        train_folds_relpath  = conf["file_locations"]["training_folds"][sampling][target_var]
        train_folds_path = [dataset_root_dir + relpath for relpath in train_folds_relpath]
        test_fold_norm_df = [None] * num_CV_folds
        train_fold_norm_df = [None] * num_CV_folds
        train_set_norm_df, test_set_norm_df = normalize_train_test(pandas.read_csv(training_set_path),
                                                                   pandas.read_csv(test_set_path), input_features,
                                                                   target_col)
        for i in range(num_CV_folds):
            train_fold_norm_df[i],test_fold_norm_df[i] = normalize_train_test(pandas.read_csv(train_folds_path[i]), pandas.read_csv(test_folds_path[i]), input_features, target_col)

        for step, mu, fil in itertools.product(wrapper_opt_step[target_var], mutual_th_ops[target_var], filt_thresholds[target_var]):
            scoring_func = globals()[step[0]]
            step_threshold = step[1]
            sorted_features = filter_and_sort_features(train_set_norm_df, input_features, target_col, fil, mu)
            column_name = "fil" + str(fil) + "_mu" + str(mu) + "_" + scoring_func.__name__ + str(step_threshold)
            # print("calculating results for " + column_name)
            if len(sorted_features)>0:
                working_model = LinearRegression()
                sel_features, score = wrapper_optimize(working_model, step, sorted_features, train_fold_norm_df, test_fold_norm_df,
                                                       target_col)
                results_df[column_name] = get_resutls_column(working_model, train_fold_norm_df, test_fold_norm_df, train_set_norm_df,
                                                             test_set_norm_df, sel_features, target_col)
            else:
                results_df[column_name] = ["", "", "", "", "", "", "", "", "", "0"] + [""]*(len(row_labels)-10)
                # best_feature

        results_df.to_csv("results_linreg_"+target_col+"_"+sampling+".csv",columns=sorted(results_df.columns))































































































































