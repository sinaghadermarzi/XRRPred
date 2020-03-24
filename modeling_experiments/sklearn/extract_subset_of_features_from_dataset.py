
import pandas
import itertools
import numpy
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics.regression import mean_absolute_error,mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler





############## read configurations
import json
with open("configurations.json") as jsonfile:
    conf = json.load(jsonfile)
dataset_root_dir = conf["dataset_root_dir"]
sampling_methods = ["nosampling"]
test_set_path = dataset_root_dir + conf["file_locations"]["test_set"]
test_folds_relpath = conf["file_locations"]["test_folds"]
test_folds_path = [dataset_root_dir + relpath for relpath in test_folds_relpath]


man_selected_features = {
    "resolution":['sum_chain_length', 'num_chains', 'frac_minc_bottom3foldunfold', 'frac_maxc_AA_G',
                  'frac_avgc_sidechainpolarity_nonpolar', 'ius_avgc_maxcons1sbin', 'ius_minc_avg', 'iul_avgc_maxcons1sbin', 'iul_minc_avg', 'frac_maxc_AA_A', 'frac_avgc_sidechainclass_aliphatic', 'frac_avgc_sidechainclass_amide', 'frac_maxc_AA_N'],

    "rfree" : ['num_chains', 'frac_avgc_bottom3foldunfold', 'frac_avgc_AA_G', 'frac_maxc_smallAGCSVTDNP', 'idx_maxc_maxswnum20_top3_bvalue', 'frac_maxc_lightGASPV', 'frac_maxc_AA_L', 'frac_avgc_hydrophob', 'frac_minc_AA_W', 'frac_maxc_sidechaincharge_negative', 'frac_minc_AA_A', 'frac_maxc_top3Flexbilityidx', 'frac_avgc_AA_A', 'frac_minc_top3disprot', 'frac_minc_top3foldunfold', 'frac_avgc_top3foldunfold', 'frac_minc_AA_N', 'ius_avgc_maxcons1sbinpl', 'ius_avgc_avg_asa_th2', 'iul_minc_avg', 'frac_minc_AA_F', 'ius_maxc_maxcons1sbin', 'iul_maxc_maxminswsum10_asa_th1', 'iul_maxc_frac1sbin_asa_th1']
}


num_CV_folds = len(test_folds_path)


short ={"resolution":"res","rfree":"rfree"}
for target_var in ["resolution","rfree"]:
    target_col = conf["target_col"][target_var]
    sampling = "nosampling"
    training_set_path = dataset_root_dir+conf["file_locations"]["training_set"][sampling][target_var]
    train_folds_relpath  = conf["file_locations"]["training_folds"][sampling][target_var]
    train_folds_path = [dataset_root_dir + relpath for relpath in train_folds_relpath]
    pandas.read_csv(test_set_path).loc[:, man_selected_features[target_var]+[target_col]].to_csv(
        "./database_with_selected_features/"+short[target_var]+"_test_set.csv", index=False)
    pandas.read_csv(training_set_path).loc[:, man_selected_features[target_var]+[target_col]].to_csv(
        "./database_with_selected_features/" + short[target_var] + "_training_set.csv", index=False)

    for i in range(num_CV_folds):
        pandas.read_csv(train_folds_path[i]).loc[:, man_selected_features[target_var]+[target_col]].to_csv(
            "./database_with_selected_features/" + short[target_var] + "_" + "training_fold"+str(i)+"_"+
            ".csv",
            index=False)
        pandas.read_csv(test_folds_path[i]).loc[:, man_selected_features[target_var]+[target_col]].to_csv(
            "./database_with_selected_features/" + short[target_var] + "_" + "test_fold"+str(i)+".csv",
            index=False)






