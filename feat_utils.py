#-*- coding: utf-8 -*-
from math import exp, log, sqrt 

#base features
def try_divide(x, y, val=0.0):
    """ 
        Try to divide two numbers
    """
    if y != 0.0:
        val = float(x) / y
    return val

def get_jaccard(seq1, seq2):
    """Compute the Jaccard distance between the two sequences `seq1` and `seq2`.
    They should contain hashable items.
    
    The return value is a float between 0 and 1, where 0 means equal, and 1 totally different.
    """
    set1, set2 = set(seq1), set(seq2)
    return 1 - len(set1 & set2) / float(len(set1 | set2))

def get_dice(A,B):
    A, B = set(A), set(B)
    intersect = len(A.intersection(B))
    union = len(A) + len(B)
    d = try_divide(2*intersect, union)
    return d

def get_sorensen(seq1, seq2):
    """Compute the Sorensen distance between the two sequences `seq1` and `seq2`.
    They should contain hashable items.
    
    The return value is a float between 0 and 1, where 0 means equal, and 1 totally different.
    """
    set1, set2 = set(seq1), set(seq2)
    return 1 - (2 * len(set1 & set2) / float(len(set1) + len(set2)))

def get_count_q1_in_q2(seq1,seq2):
    set1, set2 = set(seq1), set(seq2)
    return len(set1 & set2)

def get_ratio_q1_in_q2(seq1,seq2):
    set1, set2 = set(seq1), set(seq2)
    try:
        return len(set1 & set2)/float(len(set1))
    except:
        return 0.0

def get_count_of_sen(seq1):
    return len(seq1)

def get_count_of_unique_sen(seq1):
    set1 = set(seq1)
    return len(set1)

def get_ratio_of_unique_sen(seq1):
    set1 = set(seq1)
    try:
        return len(set1)/float(len(seq1))
    except:
        return 0.0

def get_count_of_digit(seq1):
    return sum([1. for w in seq1 if w.isdigit()])

def get_ratio_of_digit(seq1):
    try:
        return sum([1. for w in seq1 if w.isdigit()])/float(len(seq1))
    except:
        return 0.0
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

#position features
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

def base_fea(q1_ngram,q2_ngram):

    jaccard = get_jaccard(q1_ngram,q2_ngram)
    dice = get_dice(q1_ngram,q2_ngram)

    count_q1_in_q2 = get_count_q1_in_q2(q1_ngram,q2_ngram)
    ratio_q1_in_q2 = get_ratio_q1_in_q2(q1_ngram,q2_ngram)

    count_of_sen1 = get_count_of_sen(q1_ngram)
    count_of_sen2 = get_count_of_sen(q2_ngram)

    count_of_sen_min = min(count_of_sen1,count_of_sen2)
    count_of_sen_max = max(count_of_sen1,count_of_sen2)
            
    count_of_unique_sen1 = get_count_of_unique_sen(q1_ngram)
    count_of_unique_sen2 = get_count_of_unique_sen(q2_ngram)
            
    count_of_unique_sen_min = min(count_of_unique_sen1,count_of_unique_sen2)
    count_of_unique_sen_max = max(count_of_unique_sen1,count_of_unique_sen2)
            
    ratio_of_unique_sen1 = get_ratio_of_unique_sen(q1_ngram)
    ratio_of_unique_sen2 = get_ratio_of_unique_sen(q2_ngram)
            
    ratio_of_unique_sen_min = min(ratio_of_unique_sen1,ratio_of_unique_sen2)
    ratio_of_unique_sen_max = max(ratio_of_unique_sen1,ratio_of_unique_sen2)
            
    count_of_digit_sen1 = get_count_of_digit(q1_ngram)
    count_of_digit_sen2 = get_count_of_digit(q2_ngram)
                        
    count_of_digit_sen_min = min(count_of_digit_sen1,count_of_digit_sen2)
    count_of_digit_sen_max = max(count_of_digit_sen1,count_of_digit_sen2)
            
    ratio_of_digit_sen1 = get_ratio_of_digit(q1_ngram)
    ratio_of_digit_sen2 = get_ratio_of_digit(q2_ngram)
                        
    ratio_of_digit_sen_min = min(ratio_of_digit_sen1,ratio_of_digit_sen2)
    ratio_of_digit_sen_max = max(ratio_of_digit_sen1,ratio_of_digit_sen2)
            
    pos_list = get_position_list(q1_ngram,q2_ngram)
    min_pos_q1_in_q2 = min(pos_list)
    max_pos_q1_in_q2 = max(pos_list)
    mean_pos_q1_in_q2 = mean(pos_list)
    median_pos_q1_in_q2 = median(pos_list)
    std_pos_q1_in_q2 = std(pos_list)

    pos_list = get_position_list(q2_ngram,q1_ngram)
    min_pos_q2_in_q1 = min(pos_list)
    max_pos_q2_in_q1 = max(pos_list)
    mean_pos_q2_in_q1 = mean(pos_list)
    median_pos_q2_in_q1 = median(pos_list)
    std_pos_q2_in_q1 = std(pos_list)

    min_pos_q1_in_q2_1 = try_divide(min_pos_q1_in_q2,len(q1_ngram))
    max_pos_q1_in_q2_1 = try_divide(max_pos_q1_in_q2,len(q1_ngram))
    mean_pos_q1_in_q2_1 = try_divide(mean_pos_q1_in_q2,len(q1_ngram))
    median_pos_q1_in_q2_1 = try_divide(median_pos_q1_in_q2,len(q1_ngram))
    std_pos_q1_in_q2_1 = try_divide(std_pos_q1_in_q2,len(q1_ngram))

    min_pos_q2_in_q1_1 = try_divide(min_pos_q2_in_q1,len(q2_ngram))
    max_pos_q2_in_q1_1 = try_divide(max_pos_q2_in_q1,len(q2_ngram))
    mean_pos_q2_in_q1_1 = try_divide(mean_pos_q2_in_q1,len(q2_ngram))
    median_pos_q2_in_q1_1 = try_divide(median_pos_q2_in_q1,len(q2_ngram))
    std_pos_q2_in_q1_1 = try_divide(std_pos_q2_in_q1,len(q2_ngram))

    return [
        jaccard, dice,count_q1_in_q2,ratio_q1_in_q2,
        count_of_sen1,count_of_sen2,
        count_of_unique_sen1,count_of_unique_sen2,
        ratio_of_unique_sen1,ratio_of_unique_sen2,
        count_of_digit_sen1,count_of_digit_sen2,
        ratio_of_digit_sen1,ratio_of_digit_sen2,
        min_pos_q1_in_q2,max_pos_q1_in_q2,
        mean_pos_q1_in_q2,median_pos_q1_in_q2,
        std_pos_q1_in_q2,
        min_pos_q2_in_q1,max_pos_q2_in_q1,
        mean_pos_q2_in_q1,median_pos_q2_in_q1,
        std_pos_q2_in_q1,
        min_pos_q1_in_q2_1,max_pos_q1_in_q2_1,
        mean_pos_q1_in_q2_1,median_pos_q1_in_q2_1,
        std_pos_q1_in_q2_1,
        min_pos_q2_in_q1_1,max_pos_q2_in_q1_1,
        mean_pos_q2_in_q1_1,median_pos_q2_in_q1_1,
        std_pos_q2_in_q1_1,
        count_of_sen_min,count_of_sen_max,
        count_of_unique_sen_min,count_of_unique_sen_max,
        ratio_of_unique_sen_min,ratio_of_unique_sen_max,
        count_of_digit_sen_min,count_of_digit_sen_max,
        ratio_of_digit_sen_min,ratio_of_digit_sen_max,
        ]
