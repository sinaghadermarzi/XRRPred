
# used when we have a set of scores A and a target distribution B (given as a set of numbers)
# and we want to map the scores in A to the numbers in B, while preserving their order.
# to find the output for an element x, we find the element of the same relative rank within the target distibution
# the code here implements a transformer this task

import numpy
from sklearn.base import BaseEstimator, TransformerMixin




class dist_mapper(BaseEstimator, TransformerMixin):
    __sorted_src_values = None
    __sorted_trg_values = None
    # Class Constructor
    def __init__(self):
        return None

    def __rank(self, x):
        #finds the rank of the closest element to x in the arr
        arr = self.__sorted_src_values
        r = len(arr) - 1
        l = 0
        while l <= r:
            mid = l + (r - l) // 2;
            # Check if x is present at mid
            if arr[mid] == x:
                return mid
                # If x is greater, ignore left half
            elif arr[mid] < x:
                l = mid + 1
            # If x is smaller, ignore right half
            else:
                r = mid - 1
        if l == len(arr):
            return l - 1
        elif r == -1:
            return 0
        else:
            dist_l = arr[l] - x
            dist_r = x - arr[r]
            if dist_l > dist_r:
                return r
            else:
                return l

    def __relative_rank(self,x):
        #converts rank to relative rank (between 0 and 1)
        r = self.__rank(x)
        rr  = r/len(self.__sorted_src_values)
        return rr

    def fit(self, source_values, target_values):
        self.__sorted_src_values = sorted(source_values)
        self.__sorted_trg_values = sorted(target_values)
        return self

    def transform(self, X):
        X_r = list(numpy.ravel(X))
        X_transformed  = numpy.array([0] * len (X_r),dtype = float)
        for i in range(len(X_r)):
            r  = self.__relative_rank(X_r[i])
            trg_idx = int(r*len(self.__sorted_trg_values))
            X_transformed[i] = self.__sorted_trg_values[trg_idx]
        X_transformed = numpy.reshape(X_transformed,numpy.shape(X))
        return X_transformed



