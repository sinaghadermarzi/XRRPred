from residuelevel_scores import *

sequence = "MADDPSAADRNVEIWKIKKLIKSLEAARGNGTSMISLIIPPKDQISRVAKMLADEFGTASNIKSRVNRLSVLGAITSVQQRLKLYNKVPPNGLVVYCGTIVTEEGKEKKVNIDFEPFKPINTSLYLCDNKFHTEALTALLSDDSKFGFIVIDGSGALFGTLQGNTREVLHKFTVDLPKKHGRGGQSALRFARLRMEKRHNYVRKVAETAVQLFISGDKVNVAGLVLAGSADFKTELSQSDMFDQRLQSKVLKLVDISYGGENGFNQAIELSTEVLSNVKFIQEKKLIGRYFDEISQDTGKYCFGVEDTLKALEMGAVEILIVYENLDIMRYVLHCQGTEEEKILYLTPEQEKDKSHFTDKETGQEHELIESMPLLEWFANNYKKFGATLEIVTDKSQEGSQFVKGFGGIGGILRYRVDFQGMEYQGGDDEFFDLDDY"

print("profbval out:\n",profbval(sequence))

print("ASAquick out:\n",asaquick(sequence))

print("iupred short out:\n",iupred_short(sequence))

print("iupred long out:\n",iupred_long(sequence))

print("ANCHOR out:\n",ANCHOR(sequence))