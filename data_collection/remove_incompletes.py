
import csv
inputfilename = 'query30_protonly.csv'
missing_ids = set()

with open(inputfilename) as csvfile:
    reader = csv.DictReader(csvfile)
    field_names = reader.fieldnames
    for row in reader:
        pdbid = row["PDB ID"]
        seq = row["Sequence"]
        if (len(seq)<=20) or ("X" in seq) or ("U" in seq):
            print(pdbid)
            missing_ids.add(pdbid)
        # for el in row.keys():
        #     if row[el] =="":
        #         pdb_id = row["PDB ID"]
        #         missing_ids.append(pdb_id)
        #         print(pdb_id,"\t:\t", el)

outputfilename =inputfilename[:-4] +"_invalidsremoved.txt"

accepted_ids  = set()
with open(inputfilename) as csvfile:
    reader = csv.DictReader(csvfile)
    # field_names = reader.fieldnames
    # writer = csv.DictWriter(outcsv,fieldnames = field_names,delimiter = '\t')
    # writer.writeheader()
    for row in reader:
        if row['PDB ID'] not in missing_ids:
            accepted_ids.add(row['PDB ID'])
            # writer.writerow(row)
            #


with open(outputfilename,"w",newline="") as outfile:
    outfile.writelines("\n".join(accepted_ids))