import pandas as pd
from scipy.stats import norm
import numpy as np
from functools import reduce
import datetime

class User:
    
    def _aggregate_score_variable(self, variable):
        #take the mean score per day
        return variable.resample('D').mean()

    def _aggregate_binary_variable(self, variable):
        #count the occurences per day
        return variable.resample('D').sum()

    def _aggregate_duration_variable(self, variable):
        #sum the durations and count frequencies per day
        total_duration = variable.resample('D').sum()
        total_duration.rename(variable.name+'_sum', inplace = True)
        total_occ = variable.resample('D').count()
        total_occ.rename(variable.name+'_count', inplace = True)

        return (total_duration, total_occ)

    def _aggregate_variable(self, variable):
        #aggregates a variable on day level
        #takes a time series of a variable as input
        score_variables = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity']
        binary_variables = ['call', 'sms']
        #all other variables are duration variables
        if variable.name in score_variables:
            return self._aggregate_score_variable(variable)
        elif variable.name in binary_variables:
            return self._aggregate_binary_variable(variable)
        else:
            return self._aggregate_duration_variable(variable)

    def _normalize_variable(self, variable):
        #scales variable back to standard normal
        return variable.apply(lambda x: (x - variable.mean()/variable.std()))

    def _get_breakpoints(self, alphabet_size):
        #computes borders of alphabet_size equiprobable intervals for standard normal
        return norm.ppf(np.linspace(0,1,alphabet_size))
    
    def _map_to_sax(self, value, breakpoints, alphabet_size):
        #finds interval for value and returns corresponding symbol
        #map all NaNs to the same symbol
        if np.isnan(value):
            return chr(97+alphabet_size)

        interval = 0
        #pop -inf border
        breakpoints = list(breakpoints)[1:]
        for border in breakpoints:
            if value > border:
                interval += 1
            else:
                #interval found
                break
        
        return chr(97+interval)




    def _variable_to_sax(self, variable, alphabet_size):
        #don't rewrite the response variable
        if variable.name == 'mood':
            return variable
        norm_var = self._normalize_variable(variable)

        #get breakpoints
        breakpoints = self._get_breakpoints(alphabet_size)

        #map values to class/symbol
        sax_var = variable.apply(lambda x: self._map_to_sax(x, breakpoints, alphabet_size))
        return sax_var
        
    def get_sax_representation(self, alphabet_size):
        self.sax_representation = pd.DataFrame()
        self.alphabet_size = alphabet_size
        for var in self.variables:
            self.sax_representation[var] = self._variable_to_sax(self.variables[var], alphabet_size)

    def _transform_variables(self, raw_variables):
        #transform dataframe to time series per variable
        result = pd.DataFrame(index = pd.date_range(start = self.min_time.day, end = self.max_time.day, freq = 'D'))
        #result.set_index pd.date_range(start = self.min_time, end = self.max_time)
        for raw_variable in raw_variables:
            variable = raw_variable.value
            variable.rename(raw_variable.variable.iloc[0], inplace = True)
            variable.index = raw_variable.time
            #transform and save variable
            transformed_var = self._aggregate_variable(variable)
            #handle different dimensionality of transformations
            if not isinstance(transformed_var, tuple):
                transformed_var = [transformed_var]

            for var in transformed_var:
                result = pd.concat([result,var], axis = 1)
            

        return result
    def _shift_variables(self, variables, nr_days=1):
        #shifts variables over time
        result  = variables.shift(nr_days, freq='D')
        result.columns = [str(nr_days)+ col for col in result]
        return result

    def _merge_shifted_variables(self, variables, nr_days):
        #create variable dataframe with history and filter overall timeframe accordingly
        vars_with_hist = pd.concat(variables, axis=1)
        vars_with_hist = vars_with_hist.loc[((vars_with_hist.index >= self.min_time+datetime.timedelta(days=nr_days)) & (vars_with_hist.index <= self.max_time))]

        return vars_with_hist

    def _post_process_history(self, variables, use_sax):
        #clean df and return (features,response) tuple in np format
        variables = variables.dropna(subset=['mood'])
        response = variables.mood.copy()
        variables.drop([response.name], axis = 1, inplace = True)

        if use_sax:
            mood_hist_cols = [col for col in variables if ('mood' in col) and (len(col) > 4)]
            for col in mood_hist_cols:
                sax_repr =  self._variable_to_sax(variables[col], self.alphabet_size)
                variables.drop([col], axis = 1, inplace = True)
                variables[col] = sax_repr

        self.history_response = response
        self.history_features = variables

    def create_history(self, nr_days, use_sax = False):
        if use_sax: 
            variables = self.sax_representation.copy()
        else:
            variables = self.variables.copy()
        #copy response so we can add it back in later
        print(self.variables.head())
        input('hallo')
        response = variables.mood.copy()

        #initialize df with right time span
        #create list of variables shifted over desired timeframe

        shifted_variables = [variables]
        for i in range(1, nr_days):
            shifted_variables.append(self._shift_variables(variables, i))
        #print(shifted_variables)

        vars_with_hist = self._merge_shifted_variables(shifted_variables, nr_days)
        vars_with_hist.to_csv('test.csv')

        self._post_process_history(vars_with_hist, use_sax)






        


    def __init__(self, df):
        self.id = df.id.iloc[0]
        self.min_time = min(df.time)
        self.max_time = max(df.time)
        self.variables = self._transform_variables([x for _, x in df.groupby(df['variable'])])
        self.alphabet_size = None 


class Dataset:
    #Dataset containing all user information
    def __init__(self, df):

        self.users = [User(x) for _, x in df.groupby(df['id'])]
        self.nr_users = len(self.users)
        self.min_time = min(df.time)
        self.max_time = max(df.time)

