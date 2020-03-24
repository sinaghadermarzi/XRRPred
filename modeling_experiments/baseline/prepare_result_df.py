import pandas
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from scipy.stats import spearmanr
def prepare_result_df_testonly(pred_val,true_val,pred_val_name,true_val_name,labels):
    mae = mean_absolute_error(true_val, pred_val)
    mse = mean_squared_error(true_val, pred_val)
    pn, _ = pearsonr(true_val,pred_val)
    sp, _ = spearmanr(true_val,pred_val)
    ix  = ["MAE","MSE","Pearson","Spearman"," "]+list(labels)
    pred_val_col = [mae, mse, pn, sp, " "] + list(pred_val)
    true_val_col = [" "] * 5 + list(true_val)
    df = pandas.DataFrame()
    df[true_val_name] = true_val_col
    df[pred_val_name] = pred_val_col
    df.index = ix
    return df


    
    
    
