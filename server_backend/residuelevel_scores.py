import subprocess
import os
import shutil
import sys
from datetime import datetime
from random import choice
from string import digits

#folder paths without slash
this_script_directory = os.path.dirname(os.path.realpath(__file__))
iupred_dir = "/home/sghadermarzi/IUPRED"
asaquick_dir = "/home/sghadermarzi/ASAquick/bin"
ANCHOR_dir = "/home/sghadermarzi/ANCHOR"
if not os.path.exists(this_script_directory+'/tmp'):
    os.mkdir(this_script_directory+'/tmp')
temp_dir = this_script_directory+'/tmp'
num_iupred_decimal_points = 2
num_ANCHOR_decimal_points = 2
num_asaquick_decimal_points = 2



def profbval(seq):
    [0.0]*len(seq)


#
# def profbval(seq):
#     random_code= (''.join(choice(digits) for i in range(4)))
#     time_string=  datetime.now().strftime("%Y%m%d%H%M%S")
#     taskid = time_string+"_profbval_"+ random_code
#     filename = temp_dir+"/"+taskid
#     #profbval must be in the PATH
#     #save the current working directory
#     cwd  = os.getcwd()
#     #create a sequence file in the temp_dir
#     os.chdir(temp_dir)
#     with open(taskid+".seq","w",newline = "") as seqfile:
#         seqfile.writelines(">tmp\n"+seq)
#     # print("Running ProfBVAL on sequence"+filename)
#     command_1 = "/home/biomine/programs/blast-2.2.24/bin/blastpgp -j 3 -h 0.001 -d /home/biomine/programs/db/nrfilt/nrfilt -i "+filename+".seq -Q "+filename+".pssm -C "+filename+".chk -o "+filename+".blast"
#     # print(command_1)
#     os.system(command_1)
#     command_2 = "/usr/share/librg-utils-perl/blast2saf.pl fasta="+filename+".seq eSaf=1 saf="+filename+".saf "+filename+".blast"
#     # print(command_2)
#     os.system(command_2)
#     command_3 = "/usr/share/librg-utils-perl/copf.pl "+filename+".saf formatIn=saf formatOut=hssp fileOut="+filename+".hssp exeConvertSeq=/usr/bin/convert_seq"
#     # print(command_3)
#     os.system(command_3)
#     command_4 = "/usr/share/librg-utils-perl/hssp_filter.pl red=5 "+filename+".hssp fileOut="+filename+".fhssp"
#     # print(command_4)
#     os.system(command_4)
#     command_5 = "prof "+filename+".hssp fileRdb="+filename+".rdbProf"
#     # print(command_5)
#     os.system(command_5)
#     command_6 = "profbval "+filename+".seq "+filename+".rdbProf "+filename+".fhssp "+filename+".bVal 9 6"
#     # print(command_6)
#     os.system(command_6)
#     # command_7 = "profbval "+filename+".seq "+filename+".rdbProf "+filename+".fhssp "+filename+".rdbBval 9 5"
#     # os.system(command_7)
#     res = [0.0]*len(seq)
#     with open(temp_dir+"/"+taskid+".bVal") as resfile:
#         resfile.readline()
#         line = resfile.readline()
#         i = 0
#         while line:
#             line_spl = line.split("\t")
#             if len(line_spl) > 3:
#                 res[i] = float(line_spl[3])
#                 i+=1
#             line = resfile.readline()
#
#
#     # remove temp fasta file
#     for ext in [".seq",".bVal",".blast",".chk",".fhssp",".hssp",".pssm",".rdbProf",".saf"]:
#         os.remove(temp_dir+"/"+taskid+ext)
#
#
#
#     #go back to the previous working directory
#     os.chdir(cwd)
#
#     return res








def pred_iupred_long(seq):
    #save the current working directory
    cwd  = os.getcwd()
    random_code= (''.join(choice(digits) for i in range(4)))
    time_string=  datetime.now().strftime("%Y%m%d%H%M%S")
    taskid = time_string+"_iupredlong_"+ random_code
    #create a sequence file in the temp_dir
    os.chdir(iupred_dir)
    with open(temp_dir+"/"+taskid+".fasta","w",newline = "") as seqfile:
        seqfile.writelines(">tmp\n"+seq)
    p = subprocess.Popen([iupred_dir+"/iupred "+temp_dir+"/"+taskid+".fasta long"], bufsize=2048, shell=True,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
    # allow external program to work
    p.wait()
    # read the result to a string
    result_str = p.stdout.read().decode("utf-8")
    res_list = list()
    lines  = result_str.split("\n")
    for line in lines[9:]:
        line = line.rstrip('\n')
        line_cols = line.split()
        if len(line)>2:
            value = (float(line_cols[2]))
            value = round(value,num_iupred_decimal_points)
            res_list.append(value)

    #remove temp fasta file
    os.remove(temp_dir+"/"+taskid+".fasta")
    #go back to the previous working directory
    os.chdir(cwd)
    return res_list



def pred_iupred_short(seq):
    #save the current working directory
    cwd  = os.getcwd()
    random_code= (''.join(choice(digits) for i in range(4)))
    time_string=  datetime.now().strftime("%Y%m%d%H%M%S")
    taskid = time_string+"_iupredshort_"+ random_code
    #create a sequence file in the temp_dir
    os.chdir(iupred_dir)
    with open(temp_dir+"/"+taskid+".fasta","w",newline = "") as seqfile:
        seqfile.writelines(">tmp\n"+seq)
    p = subprocess.Popen([iupred_dir+"/iupred "+temp_dir+"/"+taskid+".fasta long"], bufsize=2048, shell=True,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
    # allow external program to work
    p.wait()
    # read the result to a string
    result_str = p.stdout.read().decode("utf-8")
    res_list = list()
    lines  = result_str.split("\n")
    for line in lines[9:]:
        line = line.rstrip('\n')
        line_cols = line.split()
        if len(line)>2:
            value = (float(line_cols[2]))
            value = round(value,num_iupred_decimal_points)
            res_list.append(value)

    #remove temp fasta file
    os.remove(temp_dir+"/"+taskid+".fasta")
    #go back to the previous working directory
    os.chdir(cwd)
    return res_list




def pred_asaquick(seq):
    #add asaquick to PATH
    sys.path
    sys.path.append(asaquick_dir)
    #save the current working directory
    cwd  = os.getcwd()
    #create a sequence file in the temp_dir
    os.chdir(asaquick_dir)
    random_code= (''.join(choice(digits) for i in range(4)))
    time_string=  datetime.now().strftime("%Y%m%d%H%M%S")
    taskid = time_string+"_asaquick_"+ random_code
    with open(temp_dir+"/"+taskid+".fasta","w",newline = "") as seqfile:
        seqfile.writelines(">tmp\n"+seq)
    p = os.system("./ASAquick "+temp_dir+"/"+taskid+".fasta > /dev/null 2>&1")
    #read the results from ASAquick
    fileaddress = "./asaq."+taskid+".fasta/rasaq.pred"
    f = open(fileaddress)
    res_list = list()
    line = f.readline()
    while line:
        line = line.rstrip('\n')
        line_cols = line.split(" ")
        value = float(line_cols[2])
        value  = round(value,num_asaquick_decimal_points)
        res_list.append(value)
        # res_list.append(line_cols[2])
        # use realine() to read next line
        line = f.readline()
    res_list = res_list[:-1]
    #remove temp fasta file
    # os.remove(temp_dir+"/"+taskid+".fasta")
    #remove the results from ASAquick
    shutil.rmtree("./asaq."+taskid+".fasta")
    #go back to the previous working directory
    os.chdir(cwd)
    return res_list


def pred_ANCHOR(seq):
    #save the current working directory
    cwd  = os.getcwd()
    random_code= (''.join(choice(digits) for i in range(4)))
    time_string=  datetime.now().strftime("%Y%m%d%H%M%S")
    taskid = time_string+"_iupredshort_"+ random_code
    #create a sequence file in the temp_dir
    os.chdir(ANCHOR_dir)
    with open(temp_dir+"/"+taskid+".fasta","w",newline = "") as seqfile:
        seqfile.writelines(">tmp\n"+seq)
    p = subprocess.Popen([ANCHOR_dir+"/anchor "+temp_dir+"/"+taskid+".fasta"], bufsize=2048, shell=True,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
    # allow external program to work
    p.wait()
    # read the result to a string
    result_str = p.stdout.read().decode("utf-8")
    res = list()
    lines  = result_str.split("\n")
    for line in lines:
        if not line.startswith("#"):
            line = line.rstrip('\n')
            line_cols = line.split()
            if len(line)>2:
                value = (float(line_cols[2]))
                value_bin = int(line_cols[3])
                value = round(value,num_ANCHOR_decimal_points)
                res.append(value)
    #remove temp fasta file
    os.remove(temp_dir+"/"+taskid+".fasta")
    #go back to the previous working directory
    os.chdir(cwd)
    return res





