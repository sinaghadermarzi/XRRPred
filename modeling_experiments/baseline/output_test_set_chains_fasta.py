import pandas as pd


chain_data =pd.read_csv("../../data_collection/dataset30_seq_cleaned.csv")
test_set_prots = pd.read_csv("../../dataset/original/training_set.csv").loc[:,"PDB ID"].values
with open("training_chains.fasta","w",newline= "") as outfile:
    for id in test_set_prots:
        rows = chain_data.loc[chain_data["PDB ID"]==id,:]
        for index,row in rows.iterrows():
            chainid = row["Chain ID"]
            sequence = row["Sequence"]
            out_piece = ">"+id+"-"+chainid+"\n"+sequence+"\n"
            outfile.writelines(out_piece)
