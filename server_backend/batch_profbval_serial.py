
import os

filename = str(i)
# print("Running ProfBVAL on sequence"+filename)
command_1 = "/home/biomine/programs/blast-2.2.24/bin/blastpgp -j 3 -h 0.001 -d /home/biomine/programs/db/nrfilt/nrfilt -i "+filename+".seq -Q "+filename+".pssm -C "+filename+".chk -o "+filename+".blast"
os.system(command_1)
command_2 = "/usr/share/librg-utils-perl/blast2saf.pl fasta="+filename+".seq eSaf=1 saf="+filename+".saf "+filename+".blast"
os.system(command_2)
command_3 = "/usr/share/librg-utils-perl/copf.pl "+filename+".saf formatIn=saf formatOut=hssp fileOut="+filename+".hssp exeConvertSeq=/usr/bin/convert_seq"
os.system(command_3)
command_4 = "/usr/share/librg-utils-perl/hssp_filter.pl red=5 "+filename+".hssp fileOut="+filename+".fhssp"
os.system(command_4)
command_5 = "prof "+filename+".hssp fileRdb="+filename+".rdbProf"
os.system(command_5)
command_6 = "profbval "+filename+".seq "+filename+".rdbProf "+filename+".fhssp "+filename+".bVal 9 6"
os.system(command_6)
# command_7 = "profbval "+filename+".seq "+filename+".rdbProf "+filename+".fhssp "+filename+".rdbBval 9 5"
# os.system(command_7)