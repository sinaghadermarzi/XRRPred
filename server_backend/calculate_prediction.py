from calc_prot_features import *
import os

infilename = "input_text.txt"

with open("input_text.txt") as infile:
    input_text = infile.read()

if ">" in input_text:
    number_of_chains = input_text.count(">")
    prot = [{}] * number_of_chains
    text_spl = input_text.split(">")
    if "" in text_spl:
        text_spl.remove("")
    for spl in text_spl:
        lines = spl.split("\n")[1:]
        if "" in lines:
            lines.remove("")
        seq = "".join(lines)
        prot[i]["sequence"] = seq
        prot[i]["iupred_short"] = iupred_short(seq)
        prot[i]["iupred_long"] = iupred_long(seq)
        prot[i]["profbval"] = profbval(seq)
        prot[i]["asaquick"] = asaquick(seq)
else:
    chains = input_text.split("\n")
    if "" in chains:
        chains.remove("")
    prot = [{}]*len(chains)
    for i in range(len(chains)):
            seq = chains[i]
            prot[i]["sequence"] = seq
            prot[i]["iupred_short"] = iupred_short(seq)
            prot[i]["iupred_long"] = iupred_long(seq)
            prot[i]["profbval"] = profbval(seq)
            prot[i]["asaquick"] = asaquick(seq)
    prots = [prot]



rows = []
for prot in prots:
    row = calculate_features(prot)
    rows.append(row)
out_df = pandas.DataFrame(rows)
out_df.to_csv(infilename[:-4]+"_features.txt",index = False, sep = "\t")


