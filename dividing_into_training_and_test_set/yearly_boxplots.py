import pandas as pd
import datetime as dt
from report_statistics import *
import datetime as dt


df= pd.read_csv("wholepdb_protonly_xray_res_rfree_allyears.csv")

df['Dep. Date'] = pd.to_datetime(df['Dep. Date'])

start_year = 1993
end_year = 2020
length = end_year - start_year +1
resolution = [None]*length
rfree = [None]*length
res_label = [None]*length
rfree_label = [None]*length
for y in range (start_year,end_year+1):
    idx = y-1993
    resolution[idx] = df.loc[df['Dep. Date'].dt.year == y,"Resolution"].values
    rfree[idx] = df.loc[df['Dep. Date'].dt.year == y, "R Free"].values
    res_label[idx] = str(y)+"["+str(len(resolution[idx]))+"]"
    rfree_label[idx] = str(y)+"["+str(len(rfree[idx]))+"]"

do_stat(rfree,rfree_label,name="rfree")
do_stat(resolution,rfree_label,name="resolution")