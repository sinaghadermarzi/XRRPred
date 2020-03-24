import numpy
import pandas
import seaborn
import random
from scipy.stats import pearsonr
import sys
from scipy.stats import ttest_ind
# sample_size  = 200
num_samp = 10

# in_file_name = sys.argv()[1]




in_file_name = "preds_rfree.txt"
target_col = "Native R-Free"

in_file_name = "preds_resolution.txt"
target_col = "Native Resolution"


# read in results
df = pandas.read_csv(in_file_name, sep = "\t")
num_obj,num_cols = df.shape
pred_names = list(df.columns)
pred_names.remove(target_col)
sample_size  = int(num_obj/2)
results_array = numpy.zeros((num_samp,len(pred_names)))
Y_all = df[target_col].values

for i in range(num_samp):
    # generate random array of indices (for now without replacement)
    samp_idx  = random.sample(range(num_obj),sample_size,)
    for j in range(len(pred_names)):
        X = df[pred_names[j]].values[samp_idx]
        Y = Y_all[samp_idx]
        r, p = pearsonr(X,Y)
        results_array[i,j] = r




out_file_name = in_file_name[:-4]+"_correlations"
from matplotlib import pyplot
result_df = pandas.DataFrame(results_array,index=list(range(num_samp)),columns=pred_names)
result_df.boxplot(vert=False)
pyplot.tight_layout()
pyplot.savefig(out_file_name+".png",dpi = 600)
result_df.to_csv(out_file_name+".txt",sep = "\t")

ref_set = ["random same dist","DNN++ SMOTE_ENN"]
significance_array = numpy.zeros((len(ref_set),len(result_df.columns)))
for i in range(len(ref_set)):
    ref_array  = result_df[ref_set[i]]
    for j in range(len(result_df.columns)):
        curr_col  = result_df.columns[j]
        if curr_col != ref_set[i]:
            curr_col_val = result_df[result_df.columns[j]].values
            t,p = ttest_ind(curr_col_val,ref_array)
            mean_curr  = numpy.mean(curr_col_val)
            mean_ref = numpy.mean(ref_array)
            if mean_curr < mean_ref:
                p = -p
            significance_array[i,j] = p


significance_df = pandas.DataFrame(significance_array,index=ref_set,columns=result_df.columns)
significance_df.to_csv(in_file_name[:-4]+"_significances.txt",sep = "\t")


