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


test = pd.read_csv('data/test_set_VU_DM_2014.csv')
test['ratio'] = test.price_usd / test.prop_review_score

t = time.time()

rankings = test.sort_values(['srch_id', 'ratio'])[['srch_id', 'prop_id']]
rankings.to_csv('predictions.csv', header=['SearchId', 'PropertyId'], index=None)

print('time elapsed: {}'.format(time.time() - t))
