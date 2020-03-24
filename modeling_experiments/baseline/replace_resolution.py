
import csv

resolution = {}
pred_res = {}

chains = {}

with open("blast_matched.csv") as infile, open("train_set_step4_features_imputed.txt") as resolution_file, open("similarity_based_res.txt",'w',newline='') as outfile, open("test_set_step4_features_imputed.txt") as test_set_file:
    reader = csv.DictReader(resolution_file,delimiter = '\t')
    for row in reader:
        pdbid = row["PDB ID"]
        res = row["Native_Resolution"]
        resolution[pdbid] = float(res)
    reader  = csv.DictReader(test_set_file, delimiter = '\t')
    test_ids  = []
    for row in reader:
        pdbid = row['PDB ID']
        chains[pdbid] = []
        test_ids.append(pdbid)

    line = infile.readline()
    while line:
        line = line.rstrip('\n')
        line_spl = line.split(',')
        test_pdbchain = line_spl[0]
        train_pdbchain = line_spl[1]
        if not train_pdbchain=='None':
            train_pdbchain_spl = train_pdbchain.split('-')
            train_pdb = train_pdbchain_spl[0]
            score = float(line_spl[2])
            test_pdbchain_spl = test_pdbchain.split('-')
            test_pdb = test_pdbchain_spl[0]
            chains[test_pdb].append((score,resolution[train_pdb]))
        line = infile.readline()

    for pdbid in test_ids:
        if len(chains[pdbid])==0:
            res = 'None'
        else:
            big_sim = 0
            for item in chains[pdbid]:
                (sim,reso) = item
                if sim > big_sim:
                    big_sim = sim
                    res = str(reso)
        line  = pdbid+'\t'+ res+'\n'
        outfile.writelines(line)