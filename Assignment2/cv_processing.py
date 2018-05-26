import pandas as pd 
import numpy as np
from scipy import stats 

def expand_data(cross_val_results, expand_to = 30):
    """
        function takes list of lists with cross val scores [[x_1, ..., x_2], [x_1, ..., x_n]] 
        [x_1, ..., x_n] is a list containing the cv results for one genotype
        result is  cross_val_scores + random noise to fake more observations
        
      """
    curr_num_results = len(cross_val_results[0])

    # add noise
    for val in cross_val_results:
        mu = np.mean(val)
        sigma = np.std(val)
        num_obs = expand_to - curr_num_results
        noise = list(np.random.normal(mu, sigma, num_obs))
        val.extend(noise)

    return cross_val_results


def normality_test(cross_val_results):
    """
        Performs Shapiro Wilk normality test for each model in the cross_val_results
    """
    for val in cross_val_results:
        statistic, p_val = stats.shapiro(val)
        print('p value: {}, test statistic {}'.format(p_val, statistic))

def t_test_benchmark(cross_val_results, benchmark):
    """
        Performs two sided t test on cross_val_results, be sure to take into account that it is TWO sided
    """
    for val in cross_val_results:
        statistic, p_val = stats.ttest_1samp(val, benchmark)
        print('p value: {}, test statistic {}'.format(p_val, statistic))


###################################################
# INSERT DATA HERE
###################################################

benchmark = 0.4
cv_results = [[0.35, 0.34, 0.36, 0.38, 0.32]]
cv_results = expand_data(cv_results)

# Get statistics
normality_test(cv_results)
t_test_benchmark(cv_results, benchmark)