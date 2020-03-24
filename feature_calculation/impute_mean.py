from pandas import *
import csv
import sys
import os
# from Tkinter import *
# from Tkinter import *
# import Tkinter, Tkconstants, tkFileDialog







# b = Tk()

# b.iconbitmap('favicon.ico')
# b.withdraw()
# a = str(tkFileDialog.askopenfilename(initialdir=".", title="Select feature file to be imputed", filetypes=(("CSV files", "*.csv"), ("all files", "*.*"))))
#
# inputfolder,inputfilename = os.path.split(a)
#
# working_data_file_name = str(a)

inputfilename = "test_features.txt"

inputfilename_we = inputfilename[:-4]
output_file_name = inputfilename_we + '_imputed.txt'


df= read_csv(inputfilename,na_values='None',delimiter='\t')
cf = df.fillna(df.mean())

cf.to_csv(output_file_name,index=False,sep='\t')


