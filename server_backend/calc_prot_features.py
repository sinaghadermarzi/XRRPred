
from datetime import datetime
from prot_feature_implement import *
import pandas


def asaquick(seq):
    return [1]*len(seq)
def iupred_long(seq):
    return [1]*len(seq)
def iupred_short(seq):
    return [1]*len(seq)
def profbval(seq):
    return [1]*len(seq)


def calculate_features(prot):
    prot[0]["id"]= datetime.now().strftime("%Y%m%d%H%M%S")
    row  = {}
    for f in features_to_compute:
        row[f.__name__] = f(prot)
    return row

