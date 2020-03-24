
import pandas as pd

filenames  = [ "DeepCrystal_results.csv"]



chain_preds= {}
prot_pred ={}

for f in filenames:
    df = pd.read_csv(f)
    for index ,row in df.iterrows():
        fullid  = row["Sequence ID"]
        spl = fullid.split("-")
        pdbid  = spl[0]
        chainid = spl[1]
        pred_field = row["Diffraction-quality Crystal Prediction"]
        pred = pred_field
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
out_df["deepcrystal"] = pr
out_df.to_csv("others_deepcrystal.csv",index = False)

