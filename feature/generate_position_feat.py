#-*- coding: utf-8 -*-
 
from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random,shuffle
import pickle
import sys

path = "./input/"

def try_divide(x, y, val=0.0):
    """ 
        Try to divide two numbers
    """
    if y != 0.0:
        val = float(x) / y
    return val

def get_position_list(target, obs):
    """
        Get the list of positions of obs in target
    """
    pos_of_obs_in_target = [0]
    if len(obs) != 0:
        pos_of_obs_in_target = [j for j,w in enumerate(obs, start=1) if w in target]
        if len(pos_of_obs_in_target) == 0:
            pos_of_obs_in_target = [0]
    return pos_of_obs_in_target


def mean(x):
    return sum(x)/float(len(x))

def median(x):
    len_2 = int(len(x)/2)
    return x[len_2]

def std(x):
    mean_x = mean(x)
    s = 0.0
    for xx in x:
        s+=(xx-mean_x)**2
    s/=len(x)
    s = sqrt(s)
    return s

def generate_position(path,out):
    print('generate postion,data path is',path)
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        outfile.write('min_pos_q1_in_q2,max_pos_q1_in_q2,mean_pos_q1_in_q2,median_pos_q1_in_q2,std_pos_q1_in_q2,min_pos_q2_in_q1,max_pos_q2_in_q1,mean_pos_q2_in_q1,median_pos_q2_in_q1,std_pos_q2_in_q1,min_pos_q1_in_q2_nor, max_pos_q1_in_q2_nor,mean_pos_q1_in_q2_nor,median_pos_q1_in_q2_nor,std_pos_q1_in_q2_nor,min_pos_q2_in_q1_nor,max_pos_q2_in_q1_nor,mean_pos_q2_in_q1_nor,median_pos_q2_in_q1_nor,std_pos_q2_in_q1_nor\n')
        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
           
            q1 = str(row['sen1']).lower().split(' ')
            q2 = str(row['sen2']).lower().split(' ')
            pos_list = get_position_list(q1,q2)
            min_pos_q1_in_q2 = min(pos_list)
            max_pos_q1_in_q2 = max(pos_list)
            mean_pos_q1_in_q2 = mean(pos_list)
            median_pos_q1_in_q2 = median(pos_list)
            std_pos_q1_in_q2 = std(pos_list)

            pos_list = get_position_list(q2,q1)
            min_pos_q2_in_q1 = min(pos_list)
            max_pos_q2_in_q1 = max(pos_list)
            mean_pos_q2_in_q1 = mean(pos_list)
            median_pos_q2_in_q1 = median(pos_list)
            std_pos_q2_in_q1 = std(pos_list)

            min_pos_q1_in_q2_nor = try_divide(min_pos_q1_in_q2,len(q1))
            max_pos_q1_in_q2_nor = try_divide(max_pos_q1_in_q2,len(q1))
            mean_pos_q1_in_q2_nor = try_divide(mean_pos_q1_in_q2,len(q1))
            median_pos_q1_in_q2_nor = try_divide(median_pos_q1_in_q2,len(q1))
            std_pos_q1_in_q2_nor = try_divide(std_pos_q1_in_q2,len(q1))

            min_pos_q2_in_q1_nor = try_divide(min_pos_q2_in_q1,len(q2))
            max_pos_q2_in_q1_nor = try_divide(max_pos_q2_in_q1,len(q2))
            mean_pos_q2_in_q1_nor = try_divide(mean_pos_q2_in_q1,len(q2))
            median_pos_q2_in_q1_nor = try_divide(median_pos_q2_in_q1,len(q2))
            std_pos_q2_in_q1_nor = try_divide(std_pos_q2_in_q1,len(q2))

            outfile.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (
                min_pos_q1_in_q2, max_pos_q1_in_q2,
                mean_pos_q1_in_q2,median_pos_q1_in_q2,
                std_pos_q1_in_q2,
                min_pos_q2_in_q1,max_pos_q2_in_q1,
                mean_pos_q2_in_q1,median_pos_q2_in_q1,
                std_pos_q2_in_q1,
                min_pos_q1_in_q2_nor, max_pos_q1_in_q2_nor,
                mean_pos_q1_in_q2_nor,median_pos_q1_in_q2_nor,
                std_pos_q1_in_q2_nor,
                min_pos_q2_in_q1_nor,max_pos_q2_in_q1_nor,
                mean_pos_q2_in_q1_nor,median_pos_q2_in_q1_nor,
                std_pos_q2_in_q1_nor,
                ))
            c+=1
    end = datetime.now()
    print('times:',end-start)

#generate_position(path+'train_unigram.csv',path+'train_position.csv')
#generate_position(path+'valid_unigram.csv',path+'valid_position.csv')
#generate_position(path+'test_unigram.csv',path+'test_position.csv')

