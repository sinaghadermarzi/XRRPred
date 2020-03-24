from residuelevel_scores import *
from datetime import datetime
from prot_feature_implement import *
import pandas as pd






def calculate_features(prot):
    prot[0]["id"]= datetime.now().strftime("%Y%m%d%H%M%S")
    row  = {}
    for f in features_to_compute:
        row[f.__name__] = f(prot)
    return row


chain_seqs = {}
resolution = {}
rfree = {}

chain_data = pd.read_csv("dataset30_seq_cleaned.csv")
nrows, ncols = chain_data.shape

for i in range(nrows):
    pdbid = chain_data.loc[i, "PDB ID"]
    seq = chain_data.loc[i, "PDB ID"]
    if pdbid not in resolution:
        resolution[pdbid] = chain_data.loc[i, "Refinement Resolution"]
        rfree[pdbid] = chain_data.loc[i, "R Free"]
        chain_seqs[pdbid] = [chain_data.loc[i, "Sequence"]]
    else:
        chain_seqs[pdbid].append(chain_data.loc[i, "Sequence"])

with open("test_proteins.txt") as test_prots_f, open("training_proteins.txt") as training_prots_f:
    test_prots = test_prots_f.read().split("\n")
    training_prots = training_prots_f.read().split("\n")




test_rows = [None]*len(test_prots)
for i in range(len(test_prots)):
    pdbid = test_prots[i]
    seqs = chain_seqs[pdbid]
    prot = []
    print("processing test set protein ",i,"/",len(test_rows))
    j=1
    for seq in seqs:
        chain = {}
        chain["sequence"] = seq
        chain["iupred_short"] = iupred_short(seq)
        print("iupred short done for ch",j)
        chain["iupred_long"] = iupred_long(seq)
        print("iupred long done for ch",j)
        chain["asaquick"] = asaquick(seq)
        print("asaquick done for ch",j)
        prot.append(chain)
        j+=1
    print("calculating features")
    row = calculate_features(prot)
    print("features done\n\n")
    row["OUT_resolution"] = resolution[pdbid]
    row["OUT_rfree"] = rfree[pdbid]
    row["PDB ID"] = pdbid
    test_rows[i]= row



test_df = pd.DataFrame(test_rows)
test_df.to_csv("test_features.txt",index = False, sep = "\t")







training_rows = [None]*len(training_prots)
for i in range(len(training_prots)):
    pdbid = training_prots[i]
    seqs = chain_seqs[pdbid]
    prot = []
    print("processing training set protein ",i,"/",len(training_rows))
    j=1
    for seq in seqs:
        chain = {}
        chain["sequence"] = seq
        chain["iupred_short"] = iupred_short(seq)
        print("iupred short done for ch",j)
        chain["iupred_long"] = iupred_long(seq)
        print("iupred long done for ch",j)
        chain["asaquick"] = asaquick(seq)
        print("asaquick done for ch",j)
        prot.append(chain)
        j+=1
    print("calculating features")
    row = calculate_features(prot)
    print("features done\n\n")
    row["OUT_resolution"] = resolution[pdbid]
    row["OUT_rfree"] = rfree[pdbid]
    row["PDB ID"] = pdbid
    training_rows[i]= row



training_df = pd.DataFrame(training_rows)
training_df.to_csv("training_features.txt",index = False, sep = "\t")