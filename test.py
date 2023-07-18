import numpy as np
# import scipy as sp
import sys
from fractions import Fraction
from math import gcd
from random import shuffle, seed, choice
from collections import defaultdict, deque
# import matplotlib.pyplot as plt
import time
import copy
import bisect
import itertools
import multiprocessing

def solution(task):
    print(task)

def main():
    '''sys.argv[1]'''
    # task = np.concatenate((np.genfromtxt(sys.argv[1] , delimiter=",", skip_header=1)[:,:], np.genfromtxt(sys.argv[1] , delimiter=",", skip_header=1)[:,2:]), axis=1)
    task = np.genfromtxt(sys.argv[1] , delimiter=",", skip_header=1)[:,:]
    # task[:,:2] *= 2
    print(task.shape)


    start_time = time.time()

    sol = None
    # data = [task[:task.shape[0],], task[task.shape[0]:,]]
    with multiprocessing.Pool(2) as pool:
        sol = pool.map(solution, task)
    
    # sol = solution(task)
    delta_t = (time.time() - start_time)
    m = int(delta_t)//60
    s = int(delta_t)%60
    ms = int(1000*(delta_t-m*60-s))
    print(f"{m}:{s}:{ms}")

    sol = np.asarray(sol, dtype=str)
    # print(sol)

    header = (', '.join([f'X{i+1}min, Y{i+1}min, X{i+1}max, Y{i+1}max' for i in range(len(sol[0]) // 4)])).split(', ')
    # print(header)
    sol = np.insert(sol, 0, np.asarray(header, dtype=str), axis=0)
    # print(sol)
    np.savetxt("solution.csv", sol, delimiter=",", fmt="%s")

if __name__ == '__main__':
    main()