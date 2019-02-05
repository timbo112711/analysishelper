'''
Data Transformations
Version: 1.1

This module is used for transforming your data.
'''

# Libs needed
import math
import datetime
import pandas as pd
import numpy as np
from patsylearn import PatsyTransformer

def dates(start_date, end_date):
    ''' 
    Generates a list of dates for the specified range
    start_date -> The start date 
    end_date -> The day after the end date
    ''' 
    dates = []

    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]

    for date in date_generated:
        date_list = date.strftime("%Y-%m-%d")
        dates.append(date_list)

    return dates

def adstocked_advertising(adstock_rate, data):
    '''Transforming data with applying Adstock transformations
    Calling the function on data
    Must do for each channel         
    data['Channel_Adstock'] = adstocked_advertising(0.5, data['Channel'])
    '''
    adstocked_advertising = []
    for i in range(len(data)):
        if i == 0: 
            adstocked_advertising.append(data[i])
        else:
            adstocked_advertising.append(data[i] + adstock_rate * adstocked_advertising[i-1])            
    return adstocked_advertising
    
def log_transform(df, new, old):
    '''
    Take the log if a variable and then plot both old and new
    df <- The dataframe that is being used to create Adstock variables
    new -> The new variable that is being created 
    old -> The variable you want to take the log of 
    '''
    df[new] = df[old].apply(lambda x: math.log(x, 10))

def inter_terms(data, terms):
    '''
    Generates the interaction terms between two variables 
    data -> The dataframe that is being used to create the interaction terms
    terms -> The interaction terms
        ex. "TV:Radio + TV:Facebook + Radio:Facebook" 
    '''
    terms = PatsyTransformer(terms, return_type="dataframe").fit_transform(data)

    interactions_terms = pd.DataFrame(terms)

    interactions_terms_new['dates'] = dates 

    IV_data = data.join(interactions_terms_new, how='right')
    IV_data = IV_data.drop("dates", axis=1)

    return IV_data

def dummies(data, var):
    '''
    Generates dummy variables for categorical variables
    data -> The dataframe that is being used to create the dummies
    var -> The categorical variable that the dummies are based off of 
        ex. dummy_data = dummies(media_data, media_data['cat'])
    '''
    dummies = pd.get_dummies(var)

    dummies['dates'] = dates

    data_new = data.join(dummies, how='right')
    IV_data = IV_data.drop("dates", axis=1)
