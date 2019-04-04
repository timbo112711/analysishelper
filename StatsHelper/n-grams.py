import os
import re
import time
import logging
import string
import pandas as pd
import numpy as np
# Database 
from libs.Utils.utils import print_time
# NPL/n-grams using sklearn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

@print_time
def pre_process_data(df):
    # Need to replace nan's with empty strings
    df['description2'] = df['description2'].replace('nan', '', regex=True)
    df['description1'] = df['description1'].replace('nan', '', regex=True)
    df['description'] = df['description'].replace('nan', '', regex=True)
    df['headline'] = df['headline'].replace('nan', '', regex=True)
    df['headline_part1'] = df['headline_part1'].replace('nan', '', regex=True)
    df['headline_part2'] = df['headline_part2'].replace('nan', '', regex=True)
    # Need to replace all nan's with 0's where column is an int 
    df['Avg CPC'] = df['Avg CPC'].replace('nan', 0, regex=True)
    df['CPA'] = df['CPA'].replace('nan', 0, regex=True)
    # Concat the headlines together 
    df['full_headline'] = df['headline'] + " " + df['headline_part1'] + " " + df['headline_part2']
    # Concat the ad descriptions together
    df['full_description'] = df['description'] + " " + df['description1'] + " " + df['description2']
    # Concat all headlines and descriptions together
    df['full_ad'] = df['headline'] + " " + df['headline_part1'] + " " + df['headline_part2'] + " " + df['description'] + " " + df['description1'] + " " + df['description2']
    # Lower case the columns
    df['full_headline'] = df['full_headline'].str.lower()
    df['full_description'] = df['full_description'].str.lower()
    df['full_ad'] = df['full_ad'].str.lower()
    # Need to replace '&' symbol with no space 
    df['full_headline'] = df['full_headline'].replace('&', '')
    df['full_description'] = df['full_description'].replace('&', '')
    df['full_ad'] = df['full_ad'].replace('&', '')
    # Need to replace '%' symbol with no space 
    df['full_headline'] = df['full_headline'].replace('%', '')
    df['full_description'] = df['full_description'].replace('%', '')
    df['full_ad'] = df['full_ad'].replace('%', '')
    # Need to replace '-' with no space
    df['full_headline'] = df['full_headline'].replace('-', '')
    df['full_description'] = df['full_description'].replace('-', '')
    df['full_ad'] = df['full_ad'].replace('-', ' ')
    # Re-move all punctuations in the df 
    df["full_headline"] = df['full_headline'].str.replace('[^\w\s]','')
    df["full_description"] = df['full_description'].str.replace('[^\w\s]','')
    df["full_ad"] = df['full_ad'].str.replace('[^\w\s]','')
    # Re-move all stop words from the original df
    stop = stopwords.words('english')
    df["full_headline"] = df["full_headline"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    df["full_description"] = df["full_description"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    df["full_ad"] = df["full_ad"].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

    return df

@print_time
def grams_analysis(df, column, min_range, max_range):
    df = pre_process_data(df)

    word_vectorizer = CountVectorizer(ngram_range=(min_range, max_range), analyzer='word')
    sparse_matrix = word_vectorizer.fit_transform(column)
    frequencies = sum(sparse_matrix).toarray()[0]
    # Put the results into a df
    ngram_df = pd.DataFrame(frequencies, 
                            index=word_vectorizer.get_feature_names(), 
                            columns=['Term Served'])
    # Sort
    ngram_df = ngram_df.sort_values(by='Term Served', ascending=False)
    # Set the index as a column 
    ngram_df.reset_index(level=0, inplace=True)
    # Re-arrange the columns 
    ngram_df = ngram_df.rename(columns={'index':'Term'})

    return ngram_df

def find_all_ngram_terms_in_orignial(ngram_df):
    df = pre_process_data(df)
    # For headlines 
    df['headline_ngram'] = df['full_headline'].str.findall('|'.join(ngram_df.Term)).str[0]
    df['headline_ngram_2'] = df['full_headline'].str.findall('|'.join(ngram_df.Term)).str[1]
    df['headline_ngram_3'] = df['full_headline'].str.findall('|'.join(ngram_df.Term)).str[2]
    df['headline_ngram_4'] = df['full_headline'].str.findall('|'.join(ngram_df.Term)).str[3]
    # For descriptions
    df['description_ngram'] = df['full_description'].str.findall('|'.join(ngram_df.Term)).str[0]
    df['description_ngram_2'] = df['full_description'].str.findall('|'.join(ngram_df.Term)).str[1]
    df['description_ngram_3'] = df['full_description'].str.findall('|'.join(ngram_df.Term)).str[2]
    df['description_ngram_4'] = df['full_description'].str.findall('|'.join(ngram_df.Term)).str[3]
    # For full Ads 
    df['full_ad_ngram'] = df['full_ad'].str.findall('|'.join(ngram_df.Term)).str[0]
    df['full_ad_ngram_2'] = df['full_ad'].str.findall('|'.join(ngram_df.Term)).str[1]
    df['full_ad_ngram_3'] = df['full_ad'].str.findall('|'.join(ngram_df.Term)).str[2]
    df['full_ad_ngram_4'] = df['full_ad'].str.findall('|'.join(ngram_df.Term)).str[3]
    # Fill Na's
    df = df.fillna(0)

    processed_ngrams_df = df

    return processed_ngrams_df

def headlines_sum_average_metrics():
    find_all_ngram_terms_in_orignial(ngram_df)
    # Sum the variables
    headline_sums = processed_ngrams_df.groupby('headline_ngram')["Clicks", "Impr", "CTR",
                                                "Cost", "Online Appointments", "Total Invoca Calls", 
                                                "Total Appointments (Online & Call)", "CPA"].apply(lambda x : x.astype(float).sum())

    headline2_sums = processed_ngrams_df.groupby('headline_ngram_2')["Clicks", "Impr", "CTR",
                                                    "Cost", "Online Appointments", "Total Invoca Calls", 
                                                    "Total Appointments (Online & Call)", "CPA"].apply(lambda x : x.astype(float).sum())

    headline3_sums = processed_ngrams_df.groupby('headline_ngram_3')["Clicks", "Impr", "CTR",
                                                    "Cost", "Online Appointments", "Total Invoca Calls", 
                                                    "Total Appointments (Online & Call)", "CPA"].apply(lambda x : x.astype(float).sum())

    headline4_sums = processed_ngrams_df.groupby('headline_ngram_4')["Clicks", "Impr", "CTR",
                                                    "Cost", "Online Appointments", "Total Invoca Calls", 
                                                    "Total Appointments (Online & Call)", "CPA"].apply(lambda x : x.astype(float).sum())
    # Average the variables                                                
    headline_Avg_CPC_pos = processed_ngrams_df.groupby('headline_ngram')["Avg CPC", "Avg pos"].apply(lambda x : x.astype(float).mean())

    headline2_Avg_CPC_pos = processed_ngrams_df.groupby('headline_ngram_2')["Avg CPC", "Avg pos"].apply(lambda x : x.astype(float).mean())

    headline3_Avg_CPC_pos = processed_ngrams_df.groupby('headline_ngram_3')["Avg CPC", "Avg pos"].apply(lambda x : x.astype(float).mean())

    headline4_Avg_CPC_pos = processed_ngrams_df.groupby('headline_ngram_4')["Avg CPC", "Avg pos"].apply(lambda x : x.astype(float).mean())
    # Make the index (the terms) into a column
    headline_sums['Term'] = headline_sums.index
    headline2_sums['Term'] = headline2_sums.index
    headline3_sums['Term'] = headline3_sums.index
    headline4_sums['Term'] = headline4_sums.index
    headline_Avg_CPC_pos['Term'] = headline_Avg_CPC_pos.index
    headline2_Avg_CPC_pos['Term'] = headline2_Avg_CPC_pos.index
    headline3_Avg_CPC_pos['Term'] = headline3_Avg_CPC_pos.index
    headline4_Avg_CPC_pos['Term'] = headline4_Avg_CPC_pos.index
    # Merge the sum df's with the mean df's
    headline = headline_sums.merge(headline_Avg_CPC_pos, on='Term')
    headline2 = headline2_sums.merge(headline2_Avg_CPC_pos, on='Term')
    headline3 = headline3_sums.merge(headline3_Avg_CPC_pos, on='Term')
    headline4 = headline4_sums.merge(headline4_Avg_CPC_pos, on='Term')
    # Append all of the df's together 
    headline_final_df = headline.append([headline2, headline3, headline4])
    # Map the clicks 
    clicks_mapping = dict(headline_final_df[['Term', 'Clicks']].values)
    ngram_df['Clicks'] = ngram_df.Term.map(clicks_mapping)
    # Map the impressions
    impressions_mapping = dict(headline_final_df[['Term', 'Impr']].values)
    ngram_df['Impressions'] = ngram_df.Term.map(impressions_mapping)
    # Map the Avg CPC
    avg_cpc_mapping = dict(headline_final_df[['Term', 'Avg CPC']].values)
    ngram_df['Avg CPC'] = ngram_df.Term.map(avg_cpc_mapping)
    # Map the Avg pos
    Avg_pos_mapping = dict(headline_final_df[['Term', 'Avg pos']].values)
    ngram_df['Avg pos'] = ngram_df.Term.map(Avg_pos_mapping)
    # Map the CTR
    CTR_mapping = dict(headline_final_df[['Term', 'CTR']].values)
    ngram_df['CTR'] = ngram_df.Term.map(CTR_mapping)
    # Map the cost
    cost_mapping = dict(headline_final_df[['Term', 'Cost']].values)
    ngram_df['Cost'] = ngram_df.Term.map(cost_mapping)
    # Map the Online apps
    Online_apps_mapping = dict(headline_final_df[['Term', 'Online Appointments']].values)
    ngram_df['Online Appointments'] = ngram_df.Term.map(Online_apps_mapping)
    # Map the Total Invoca Calls
    Total_Invoca_Calls_mapping = dict(headline_final_df[['Term', 'Total Invoca Calls']].values)
    ngram_df['Total Invoca Calls'] = ngram_df.Term.map(Total_Invoca_Calls_mapping)
    # Map the Total Appointments (Online & Call)
    Total_Appointments_Online_Call_mapping = dict(headline_final_df[['Term', 'Total Appointments (Online & Call)']].values)
    ngram_df['Total Appointments (Online & Call)'] = ngram_df.Term.map(Total_Appointments_Online_Call_mapping)
    # Map the CPA
    CPA_mapping = dict(headline_final_df[['Term', 'CPA']].values)
    ngram_df['CPA'] = ngram_df.Term.map(CPA_mapping)
    # Fill NaN's
    ngram_df = ngram_df.fillna(0)

    return ngram_df

def descriptions_sum_average_metrics():
    find_all_ngram_terms_in_orignial(ngram_df)
    # Sum the variables
    descriptions_sums = processed_ngrams_df.groupby('description_ngram')["Clicks", "Impr", "CTR",
                                                        "Cost", "Online Appointments", "Total Invoca Calls", 
                                                        "Total Appointments (Online & Call)", "CPA"].apply(lambda x : x.astype(float).sum())

    descriptions2_sums = processed_ngrams_df.groupby('description_ngram_2')["Clicks", "Impr", "CTR",
                                                           "Cost", "Online Appointments", "Total Invoca Calls", 
                                                           "Total Appointments (Online & Call)", "CPA"].apply(lambda x : x.astype(float).sum())

    descriptions3_sums = processed_ngrams_df.groupby('description_ngram_3')["Clicks", "Impr", "CTR",
                                                           "Cost", "Online Appointments", "Total Invoca Calls", 
                                                           "Total Appointments (Online & Call)", "CPA"].apply(lambda x : x.astype(float).sum())

    descriptions4_sums = processed_ngrams_df.groupby('description_ngram_4')["Clicks", "Impr", "CTR",
                                                           "Cost", "Online Appointments", "Total Invoca Calls", 
                                                           "Total Appointments (Online & Call)", "CPA"].apply(lambda x : x.astype(float).sum())
    # Average the variables 
    descriptions_Avg_CPC_pos = processed_ngrams_df.groupby('description_ngram')["Avg CPC", "Avg pos"].apply(lambda x : x.astype(float).mean())

    descriptions2_Avg_CPC_pos = processed_ngrams_df.groupby('description_ngram_2')["Avg CPC", "Avg pos"].apply(lambda x : x.astype(float).mean())

    descriptions3_Avg_CPC_pos = processed_ngrams_df.groupby('description_ngram_3')["Avg CPC", "Avg pos"].apply(lambda x : x.astype(float).mean())

    descriptions4_Avg_CPC_pos = processed_ngrams_df.groupby('description_ngram_4')["Avg CPC", "Avg pos"].apply(lambda x : x.astype(float).mean())
    # Make the index (the terms) into a column
    descriptions_sums['Term'] = descriptions_sums.index
    descriptions2_sums['Term'] = descriptions2_sums.index
    descriptions3_sums['Term'] = descriptions3_sums.index
    descriptions4_sums['Term'] = descriptions4_sums.index
    descriptions_Avg_CPC_pos['Term'] = descriptions_Avg_CPC_pos.index
    descriptions2_Avg_CPC_pos['Term'] = descriptions2_Avg_CPC_pos.index
    descriptions3_Avg_CPC_pos['Term'] = descriptions3_Avg_CPC_pos.index
    descriptions4_Avg_CPC_pos['Term'] = descriptions4_Avg_CPC_pos.index
    # Merge the sum df's with the mean df's
    descriptions = descriptions_sums.merge(descriptions_Avg_CPC_pos, on='Term')
    descriptions2 = descriptions2_sums.merge(descriptions2_Avg_CPC_pos, on='Term')
    descriptions3 = descriptions3_sums.merge(descriptions3_Avg_CPC_pos, on='Term')
    descriptions4 = descriptions4_sums.merge(descriptions4_Avg_CPC_pos, on='Term')
    # Append all of the df's together 
    descriptions_final_df = descriptions.append([descriptions2, descriptions3, descriptions4])
    # Map the clicks 
    clicks_mapping = dict(descriptions_final_df[['Term', 'Clicks']].values)
    ngram_df['Clicks'] = ngram_df.Term.map(clicks_mapping)
    # Map the impressions
    impressions_mapping = dict(descriptions_final_df[['Term', 'Impr']].values)
    ngram_df['Impressions'] = ngram_df.Term.map(impressions_mapping)
    # Map the Avg CPC
    avg_cpc_mapping = dict(descriptions_final_df[['Term', 'Avg CPC']].values)
    ngram_df['Avg CPC'] = ngram_df.Term.map(avg_cpc_mapping)
    # Map the Avg pos
    Avg_pos_mapping = dict(descriptions_final_df[['Term', 'Avg pos']].values)
    ngram_df['Avg pos'] = ngram_df.Term.map(Avg_pos_mapping)
    # Map the CTR
    CTR_mapping = dict(descriptions_final_df[['Term', 'CTR']].values)
    ngram_df['CTR'] = ngram_df.Term.map(CTR_mapping)
    # Map the cost
    cost_mapping = dict(descriptions_final_df[['Term', 'Cost']].values)
    ngram_df['Cost'] = ngram_df.Term.map(cost_mapping)
    # Map the Online apps
    Online_apps_mapping = dict(descriptions_final_df[['Term', 'Online Appointments']].values)
    ngram_df['Online Appointments'] = ngram_df.Term.map(Online_apps_mapping)
    # Map the Total Invoca Calls
    Total_Invoca_Calls_mapping = dict(descriptions_final_df[['Term', 'Total Invoca Calls']].values)
    ngram_df['Total Invoca Calls'] = ngram_df.Term.map(Total_Invoca_Calls_mapping)
    # Map the Total Appointments (Online & Call)
    Total_Appointments_Online_Call_mapping = dict(descriptions_final_df[['Term', 'Total Appointments (Online & Call)']].values)
    ngram_df['Total Appointments (Online & Call)'] = ngram_df.Term.map(Total_Appointments_Online_Call_mapping)
    # Map the CPA
    CPA_mapping = dict(descriptions_final_df[['Term', 'CPA']].values)
    ngram_df['CPA'] = ngram_df.Term.map(CPA_mapping)
    # Fill NaN's
    ngram_df = ngram_df.fillna(0)

    return ngram_df

def full_ad_sum_average_metrics():
    find_all_ngram_terms_in_orignial(ngram_df)
    # Sum the variables
    full_ad_sums = processed_ngrams_df.groupby('full_ad_ngram')["Clicks", "Impr", "CTR",
                                                "Cost", "Online Appointments", "Total Invoca Calls", 
                                                "Total Appointments (Online & Call)", "CPA"].apply(lambda x : x.astype(float).sum())

    full_ad2_sums = processed_ngrams_df.groupby('full_ad_ngram_2')["Clicks", "Impr", "CTR",
                                                    "Cost", "Online Appointments", "Total Invoca Calls", 
                                                    "Total Appointments (Online & Call)", "CPA"].apply(lambda x : x.astype(float).sum())

    full_ad3_sums = processed_ngrams_df.groupby('full_ad_ngram_3')["Clicks", "Impr", "CTR",
                                                    "Cost", "Online Appointments", "Total Invoca Calls", 
                                                    "Total Appointments (Online & Call)", "CPA"].apply(lambda x : x.astype(float).sum())

    full_ad4_sums = processed_ngrams_df.groupby('full_ad_ngram_4')["Clicks", "Impr", "CTR",
                                                    "Cost", "Online Appointments", "Total Invoca Calls", 
                                                    "Total Appointments (Online & Call)", "CPA"].apply(lambda x : x.astype(float).sum())
    # Average the variables 
    full_ad_Avg_CPC_pos = processed_ngrams_df.groupby('full_ad_ngram')["Avg CPC", "Avg pos"].apply(lambda x : x.astype(float).mean())

    full_ad2_Avg_CPC_pos = processed_ngrams_df.groupby('full_ad_ngram_2')["Avg CPC", "Avg pos"].apply(lambda x : x.astype(float).mean())

    full_ad3_Avg_CPC_pos = processed_ngrams_df.groupby('full_ad_ngram_3')["Avg CPC", "Avg pos"].apply(lambda x : x.astype(float).mean())

    full_ad4_Avg_CPC_pos = processed_ngrams_df.groupby('full_ad_ngram_4')["Avg CPC", "Avg pos"].apply(lambda x : x.astype(float).mean())
    # Make the index (the terms) into a column
    full_ad_sums['Term'] = full_ad_sums.index
    full_ad2_sums['Term'] = full_ad2_sums.index
    full_ad3_sums['Term'] = full_ad3_sums.index
    full_ad4_sums['Term'] = full_ad4_sums.index
    full_ad_Avg_CPC_pos['Term'] = full_ad_Avg_CPC_pos.index
    full_ad2_Avg_CPC_pos['Term'] = full_ad2_Avg_CPC_pos.index
    full_ad3_Avg_CPC_pos['Term'] = full_ad3_Avg_CPC_pos.index
    full_ad4_Avg_CPC_pos['Term'] = full_ad4_Avg_CPC_pos.index
    # Merge the sum df's with the mean df's
    full_ad = full_ad_sums.merge(full_ad_Avg_CPC_pos, on='Term')
    full_ad2 = full_ad2_sums.merge(full_ad2_Avg_CPC_pos, on='Term')
    full_ad3 = full_ad3_sums.merge(full_ad3_Avg_CPC_pos, on='Term')
    full_ad4 = full_ad4_sums.merge(full_ad4_Avg_CPC_pos, on='Term')
    # Merge the sum df's with the mean df's
    full_ad_final_df = full_ad.append([full_ad2, full_ad3, full_ad4])
    # Map the clicks 
    clicks_mapping = dict(full_ad_final_df[['Term', 'Clicks']].values)
    ngram_df['Clicks'] = ngram_df.Term.map(clicks_mapping)
    # Map the impressions
    impressions_mapping = dict(full_ad_final_df[['Term', 'Impr']].values)
    ngram_df['Impressions'] = ngram_df.Term.map(impressions_mapping)
    # Map the Avg CPC
    avg_cpc_mapping = dict(full_ad_final_df[['Term', 'Avg CPC']].values)
    ngram_df['Avg CPC'] = ngram_df.Term.map(avg_cpc_mapping)
    # Map the Avg pos
    Avg_pos_mapping = dict(full_ad_final_df[['Term', 'Avg pos']].values)
    ngram_df['Avg pos'] = ngram_df.Term.map(Avg_pos_mapping)
    # Map the CTR
    CTR_mapping = dict(full_ad_final_df[['Term', 'CTR']].values)
    ngram_df['CTR'] = ngram_df.Term.map(CTR_mapping)
    # Map the cost
    cost_mapping = dict(full_ad_final_df[['Term', 'Cost']].values)
    ngram_df['Cost'] = ngram_df.Term.map(cost_mapping)
    # Map the Online apps
    Online_apps_mapping = dict(full_ad_final_df[['Term', 'Online Appointments']].values)
    ngram_df['Online Appointments'] = ngram_df.Term.map(Online_apps_mapping)
    # Map the Total Invoca Calls
    Total_Invoca_Calls_mapping = dict(full_ad_final_df[['Term', 'Total Invoca Calls']].values)
    ngram_df['Total Invoca Calls'] = ngram_df.Term.map(Total_Invoca_Calls_mapping)
    # Map the Total Appointments (Online & Call)
    Total_Appointments_Online_Call_mapping = dict(full_ad_final_df[['Term', 'Total Appointments (Online & Call)']].values)
    ngram_df['Total Appointments (Online & Call)'] = ngram_df.Term.map(Total_Appointments_Online_Call_mapping)
    # Map the CPA
    CPA_mapping = dict(full_ad_final_df[['Term', 'CPA']].values)
    ngram_df['CPA'] = ngram_df.Term.map(CPA_mapping)
    # Fill NaN's
    ngram_df = ngram_df.fillna(0)

    return ngram_df
