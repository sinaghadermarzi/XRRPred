
import pandas as pd

filenames  = [ "results_p1.csv","results_p2.csv","results_p3.csv","results_p4.csv","results_p5.csv"]



chain_preds= {}
prot_pred ={}

for f in filenames:
    df = pd.read_csv(f)
    for index ,row in df.iterrows():
        fullid  = row["Protein ID"]
        spl = fullid.split("_")
        pdbid  = spl[0]
        chainid = spl[1]
        pred_field = row[" diffraction-quality crystallization success propensity (score level)"]
        pred = float(pred_field.split("(")[0].rstrip())
        if pdbid in chain_preds:
            if chainid in chain_preds[pdbid]:
                print("error")
            chain_preds[pdbid][chainid] = pred
        else:
            chain_preds[pdbid] = {chainid:pred}



import numpy
for pdbid in chain_preds:
    prot_pred[pdbid] = numpy.mean(list(chain_preds[pdbid].values()))


mean_imp = numpy.mean(list(prot_pred.values()))
test_df = pd.read_csv("../../../dataset/original/test_set.csv")

ids = test_df["PDB ID"].values
pr = [mean_imp]*len(ids)


for i in range(len(ids)):
    pdbid = ids[i]
    pr[i] = 1-prot_pred[pdbid]

out_df = pd.DataFrame()
out_df["PDB ID"] = ids
out_df["fdetect"] = pr
out_df.to_csv("others.csv",index = False)

