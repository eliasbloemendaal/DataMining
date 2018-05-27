import pandas as pd
from datetime import datetime
import random, math
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import math
import random
import numpy as np
import time


def get_relevance(row):
    if row.booking_bool == 1:
        return 5
    elif row.click_bool == 1:
        return 1
    else:
        return 0


def get_dcg(relevances):
    dcg = 0   
    for i, rel in enumerate(relevances):
        dcg += (2 ** rel - 1) / math.log(i+2, 2)
        
    return dcg
    

def get_ndcg(relevances):
    ndcg = get_dcg(relevances) / get_dcg(relevances.sort_values(ascending=False))
    return ndcg



df = pd.read_csv('data/training_set_VU_DM_2014.csv')
df['relevance'] = df.apply(lambda row: get_relevance(row), axis=1)
df['ratio'] = df.price_usd / df.prop_review_score

t = time.time()

result = df.sort_values('ratio').groupby('srch_id').relevance.agg(get_ndcg)

print('time elapsed: {}'.format(time.time() - t))
print('mean ndcg'.format(result.mean())) 