import numpy
import pandas
import seaborn
import random
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import sys
from scipy.stats import ttest_ind
from matplotlib import pyplot
import itertools

def pvalfunc_ttestind(X,Y):
    t,p = ttest_ind(X,Y)
    return p


def scorefunc_spearman(X,Y):
    r,p = spearmanr(X,Y)
    return r


def scorefunc_pearson(X,Y):
    r,p = pearsonr(X,Y)
    return r


def scorefunc_mse(X,Y):
    return mean_squared_error(X,Y)
def scorefunc_mae(X,Y):
    return mean_absolute_error(X,Y)

def sampled_stat(data,score_func,target_col_name,num_samples,sample_size):
    num_obj, num_cols = data.shape
    samp_size = None
    if type(sample_size) == float:
        samp_size = int(sample_size*num_obj)
    elif type(sample_size) == int:
        samp_size = sample_size
    num_samp=num_samples
    pred_names = list(data.columns)
    pred_names.remove(target_col_name)
    results_array = numpy.zeros((num_samp, len(pred_names)))
    Y_all = data[target_col_name].values
    for i in range(num_samp):
        # generate random array of indices (for now without replacement)
        samp_idx = random.sample(range(num_obj), samp_size )
        for j in range(len(pred_names)):
            X = df[pred_names[j]].values[samp_idx]
            Y = Y_all[samp_idx]
            v = score_func(X, Y)
            results_array[i, j] = v
    result_df = pandas.DataFrame(results_array, index=None, columns=pred_names)
    return result_df



def bootstrapped_stat(data,score_func,target_col_name,num_samples,sample_size):
    num_obj, num_cols = data.shape
    samp_size = None
    if type(sample_size) == float:
        samp_size = int(sample_size*num_obj)
    elif type(sample_size) == int:
        samp_size = sample_size
    num_samp=num_samples
    pred_names = list(data.columns)
    pred_names.remove(target_col_name)
    results_array = numpy.zeros((num_samp, len(pred_names)))
    Y_all = data[target_col_name].values
    for i in range(num_samp):
        # generate random array of indices (for now without replacement)
        samp_idx = []
        for i in range(samp_size):
            samp_idx.append(random.sample(range(num_obj),1)[0])
        for j in range(len(pred_names)):
            X = df[pred_names[j]].values[samp_idx]
            Y = Y_all[samp_idx]
            v = score_func(X, Y)
            results_array[i, j] = v
    result_df = pandas.DataFrame(results_array, index=None, columns=pred_names)
    return result_df




def stat_sigtest(data,refs,pvalfunc):
    ref_set = refs
    significance_array = numpy.ones((len(ref_set), len(data.columns)))
    for i in range(len(ref_set)):
        ref_array = data[ref_set[i]]
        for j in range(len(data.columns)):
            curr_col = data.columns[j]
            if curr_col != ref_set[i]:
                curr_col_val = data[data.columns[j]].values
                p = pvalfunc(curr_col_val, ref_array)
                mean_curr = numpy.mean(curr_col_val)
                mean_ref = numpy.mean(ref_array)
                if mean_curr < mean_ref:
                    p = -p
                significance_array[i, j] = p

    significance_df = pandas.DataFrame(significance_array, index=ref_set, columns=result_df.columns)
    return significance_df



num_samples = 10
arg = "resolution"
arg = "rfree"
if len(sys.argv)>1:
    arg = sys.argv[1]
if arg == "resolution":
    in_file_name = "preds_resolution.txt"
    target_col = "Native Resolution"
elif arg == "rfree":
    target_col = "Native R-Free"
    in_file_name = "preds_rfree.txt"

ref_set = ["random same dist","DNN++ SMOTE_ENN"]


# read in data
df = pandas.read_csv(in_file_name, sep = "\t")
scorefuncs = [scorefunc_mae,scorefunc_mse,scorefunc_pearson,scorefunc_spearman]
sampling_methods  = [bootstrapped_stat,sampled_stat]
pvalfuncs = [pvalfunc_ttestind]

n = 0
for (scorefunc,samp_method,pvalfunc) in itertools.product(scorefuncs,sampling_methods ,pvalfuncs):
    n+=1
    result_df =  sampled_stat(df,scorefunc,target_col,num_samples,0.5)
    pyplot.figure(n)
    result_df.boxplot(vert=False)
    pyplot.tight_layout()
    pyplot.savefig(in_file_name[:-4]+"_"+ samp_method.__name__+"_" + scorefunc.__name__+".png",dpi = 600)
    result_df.to_csv(in_file_name[:-4]+"_"+ samp_method.__name__+"_" + scorefunc.__name__+".txt",sep = "\t")
    significance_df = stat_sigtest(result_df,ref_set,pvalfunc)
    significance_df.to_csv(in_file_name[:-4]+"_"+ samp_method.__name__ +"_" + scorefunc.__name__ + "_"+pvalfunc.__name__+ ".txt",sep = "\t")


