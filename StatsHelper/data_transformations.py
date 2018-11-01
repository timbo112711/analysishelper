# Libs needed 
import pandas as pd
import numpy as np

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
    '''Take the log if a variable and then plot both old and new'''
    df[new] = np.log10(df[old])

    df[old].apply(np.log).hist()
    df[new].apply(np.log).hist()
    plt.show()