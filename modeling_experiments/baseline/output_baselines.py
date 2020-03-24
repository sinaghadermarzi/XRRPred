
import random
import pandas
from prepare_result_df import *
from map_dist import *
import numpy
from sklearn.preprocessing import minmax_scale

target_name = "OUT_rfree"
test_set_df  = pandas.read_csv("../../dataset/original/test_set.csv")
train_set_df  = pandas.read_csv("../../dataset/original/training_set.csv")

# target_name = "OUTPUT_R.Free"
# test_set_df  = pandas.read_csv("../datasets/unsampled/r-free/test_set_rfree.txt",sep = "\t")
# train_set_df  = pandas.read_csv("../datasets/unsampled/r-free/train_set_rfree.txt",sep = "\t")
#
#



test_true_vals = test_set_df[target_name].values
train_true_vals = train_set_df[target_name].values

labels = test_set_df["PDB ID"].values

min_true_value = min(test_true_vals)
max_true_value = max(test_true_vals)


output_df  = pandas.DataFrame()
other_tools_df = pandas.read_csv("others.csv")

#random in range
raw_scores  = numpy.random.rand(len(test_true_vals))
pred_vals =  minmax_scale(raw_scores,(min_true_value,max_true_value))
res = prepare_result_df_testonly(pred_vals, test_true_vals, "random in range",target_name,labels)
output_df = pandas.concat([output_df,res],axis=1)


#random in same dist
tr = dist_mapper()
tr.fit(raw_scores,train_true_vals)
pred_vals = tr.transform(raw_scores)
res = prepare_result_df_testonly(pred_vals, test_true_vals, "random same dist",target_name,labels)
res = res.drop([target_name],axis = 1)
output_df = pandas.concat([output_df,res],axis=1)

# #chrysalis1 in range
# raw_scores  = other_tools_df["chrysalis1 (1-output)"]
# pred_vals =  minmax_scale(raw_scores,(min_true_value,max_true_value))
# res = prepare_result_df_testonly(pred_vals, test_true_vals, "chrysalis1 (1-output) in range",target_name,labels)
# res = res.drop([target_name],axis = 1)
# output_df = pandas.concat([output_df,res],axis=1)
#
#
# #chrysalis1 in same dist
# tr = dist_mapper()
# tr.fit(raw_scores,train_true_vals)
# pred_vals = tr.transform(raw_scores)
# res = prepare_result_df_testonly(pred_vals, test_true_vals, "chrysalis1 (1-output) same dist",target_name,labels)
# res = res.drop([target_name],axis = 1)
# output_df = pandas.concat([output_df,res],axis=1)
#
#
# #chrysalis2 in range
# raw_scores  = other_tools_df["chrysalis2 (1-output)"]
# pred_vals =  minmax_scale(raw_scores,(min_true_value,max_true_value))
# res = prepare_result_df_testonly(pred_vals, test_true_vals, "chrysalis2 (1-output) in range",target_name,labels)
# res = res.drop([target_name],axis = 1)
# output_df = pandas.concat([output_df,res],axis=1)
#
#
# #chrysalis2 in same dist
# tr = dist_mapper()
# tr.fit(raw_scores,train_true_vals)
# pred_vals = tr.transform(raw_scores)
# res = prepare_result_df_testonly(pred_vals, test_true_vals, "chrysalis2 (1-output) same dist",target_name,labels)
# res = res.drop([target_name],axis = 1)
# output_df = pandas.concat([output_df,res],axis=1)


#fdetect in range
raw_scores  = other_tools_df["fdetect"]
pred_vals =  minmax_scale(raw_scores,(min_true_value,max_true_value))
res = prepare_result_df_testonly(pred_vals, test_true_vals, "fdetect minmax",target_name,labels)
res = res.drop([target_name],axis = 1)
output_df = pandas.concat([output_df,res],axis=1)


#fdetect in same dist
tr = dist_mapper()
tr.fit(raw_scores,train_true_vals)
pred_vals = tr.transform(raw_scores)
res = prepare_result_df_testonly(pred_vals, test_true_vals, "fdetect same dist",target_name,labels)
res = res.drop([target_name],axis = 1)
output_df = pandas.concat([output_df,res],axis=1)


#fdetect in range
raw_scores  = other_tools_df["deepcrystal"]
pred_vals =  minmax_scale(raw_scores,(min_true_value,max_true_value))
res = prepare_result_df_testonly(pred_vals, test_true_vals, "deepcrystal minmax",target_name,labels)
res = res.drop([target_name],axis = 1)
output_df = pandas.concat([output_df,res],axis=1)


#fdetect in same dist
tr = dist_mapper()
tr.fit(raw_scores,train_true_vals)
pred_vals = tr.transform(raw_scores)
res = prepare_result_df_testonly(pred_vals, test_true_vals, "deepcrystal same dist",target_name,labels)
res = res.drop([target_name],axis = 1)
output_df = pandas.concat([output_df,res],axis=1)


output_df.to_csv("baselines_"+target_name+".txt",sep = "\t")