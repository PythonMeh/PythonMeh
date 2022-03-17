# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 09:40:00 2022

@author: jaguiarh
"""

#%% Import and functions
import os, time
import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd. set_option('display.max_rows', 1000)

#%% Import and functions
def get_null_perc(df): 

    '''
    for a given df, it returns a dataframe with the number and percent of missing values
    '''
    df_info = df.isnull().sum(axis=0).reset_index(name='count_size')
    df_info.count_size = df.shape[0] - df_info.count_size
    df_info['Percent'] = df_info.count_size * 100 / df.shape[0]
    df_info = df_info.sort_values(by='count_size', ascending=False)
    return df_info
#%% Import and functions
def get_all_data_in_folder(folder):
    '''
    for a given folder, it recovers all csv files in subdirectories and generates a single DataFrame with all data rows concatenated
    '''
    ddss = []
    for path, subdirs, files in os.walk(folder):
        for name in files:
            #print(os.path.join(path, name))
            temp = pd.read_csv(os.path.join(path, name), sep=";", parse_dates=[0], infer_datetime_format=True, dayfirst=True)
            if not temp.empty:
                temp['name_file'] = name
                ddss.append(temp)

    return pd.concat([d for d in ddss], axis=0, ignore_index=True)
#%% Import and functions
def clean_df(df):
    cols=[ 'COURANT_C1', 'COURANT_C2',  'COURANT_C3', 'COURANT_C4', 'COURANT_C5', 
          'PHI_C1',  'PHI_C2', 'PHI_C3', 'PHI_C4', 'PHI_C5',
          'TENSION_C1','TENSION_C2', 'TENSION_C3', 'TENSION_C4', 'TENSION_C5',
          
          'COURANT_J1','COURANT_J2', 'COURANT_J3','COURANT_J4', 'COURANT_J5', 'COURANT_J6', 'COURANT_J7', 'COURANT_J8', 'COURANT_J9','COURANT_J10',
          'COURANT_STAB1','COURANT_STAB2',   'TENSION_STAB1', 'TENSION_STAB2',
          
          'TENSION_J1','TENSION_J2', 'TENSION_J3', 'TENSION_J4', 'TENSION_J5', 'TENSION_J6', 'TENSION_J7', 'TENSION_J8', 'TENSION_J9', 'TENSION_J10', 
          'UDEMAG_J1', 'UDEMAG_J2', 'UDEMAG_J3', 'UDEMAG_J4','UDEMAG_J5', 'UDEMAG_J6', 'UDEMAG_J7', 'UDEMAG_J8', 'UDEMAG_J9', 'UDEMAG_J10', 
          
          'NOMBRE_SPIRE_REEL', 'NUMERO_BASSINE', 'TEMPSAJUST', 'TEMPSCTRL', 'NUMERO_PALETTE', 'NUMERO_STATION_AJUSTAGE', 'NB_PASSAGES','NUMERO_LIGNE',  'NUMERO_AXE_ETAMAGE']
    # Filter REFERENCE (we keep only 0Z5117 and 0Z6117)
    dout = df[df.RESULTCTRL.isin(['ZMOY', 'Zmax','Zmin','ZHAUT','ZBAS'])]
    dout = df[df.REFERENCE.isin(['0Z6117', '0Z5117'])]

    # Drop useless columns (all those with C_6 to C_20)
    list_to_drop = [
        'COURANT_C6', 'COURANT_C7', 'COURANT_C8', 'COURANT_C9', 'COURANT_C10', 'COURANT_C11','COURANT_C12',
        'COURANT_C13', 'COURANT_C14', 'COURANT_C15', 'COURANT_C16', 'COURANT_C17', 'COURANT_C18', 'COURANT_C19', 'COURANT_C20',

        'TENSION_C6', 'TENSION_C7', 'TENSION_C8', 'TENSION_C9', 'TENSION_C10', 'TENSION_C11', 'TENSION_C12',
        'TENSION_C13', 'TENSION_C14', 'TENSION_C15', 'TENSION_C16', 'TENSION_C17', 'TENSION_C18', 'TENSION_C19', 'TENSION_C20',
        
        'PHI_C6', 'PHI_C7', 'PHI_C8', 'PHI_C9', 'PHI_C10', 'PHI_C11', 'PHI_C12',
        'PHI_C13', 'PHI_C14', 'PHI_C15', 'PHI_C16', 'PHI_C17', 'PHI_C18', 'PHI_C19', 'PHI_C20'
    ]
    
    dout = dout.drop(list_to_drop, axis = 1)
    
    #Remove outliers
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_out =  dout[~((dout[cols] < (Q1 - 1.5 * IQR)) |(dout[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

    return df_out
#%% Import and functions
def calculate_Z(x):
    '''
    x[0] will be TENSION
    x[1] will be COURANT
    '''
    tension = x[0]
    courant = x[1]
    impedance = 0
    if courant == 0:
        if tension != 0:
            impedance = np.nan
    else:
        if tension < 0:
            impedance = np.nan
        else:
            impedance = tension/courant
    
    return impedance
#%% Import and functions

def calculate_columns_Z_and_S(df_):
    #Z contoLe = 
    df_['Zini']=  df_[['TENSION_C1' , 'COURANT_C1']].apply(calculate_Z, axis = 1)
    df_['Z_ctr1']=  df_[['TENSION_C2','COURANT_C2']].apply(calculate_Z, axis = 1)
    df_['Z_ctr2']=  df_[['TENSION_C3','COURANT_C3']].apply(calculate_Z, axis = 1)
    df_['Z_ctr3']=  df_[['TENSION_C4','COURANT_C4']].apply(calculate_Z, axis = 1)
    df_['Z_ctr4']=  df_[['TENSION_C5','COURANT_C5']].apply(calculate_Z, axis = 1)
    df_['Z_moy'] = (df_['Zini']+ df_['Z_ctr1']+ df_['Z_ctr2']+ df_['Z_ctr3']+ df_['Z_ctr4'])/5.0
    
    df_['Z_stab1']=  df_[['TENSION_STAB1','COURANT_STAB1']].apply(calculate_Z, axis = 1)
    df_['Z_stab2']=  df_[['TENSION_STAB2','COURANT_STAB2']].apply(calculate_Z, axis = 1)

    df_['Sini']=  df_['TENSION_C1'] * df_['COURANT_C1']
    df_['S_ctr1']=  df_['TENSION_C2'] * df_['COURANT_C2']
    df_['S_ctr2']=  df_['TENSION_C3'] * df_['COURANT_C3']
    df_['S_ctr3']=  df_['TENSION_C4'] * df_['COURANT_C4']
    df_['S_ctr4']=  df_['TENSION_C5'] * df_['COURANT_C5']
    df_['S_moy'] = (df_['Sini']+ df_['S_ctr1']+ df_['S_ctr2']+ df_['S_ctr3']+ df_['S_ctr4'])/5.0
    
    df_['S_stab1']=  df_['TENSION_STAB1'] * df_['COURANT_STAB1']
    df_['S_stab2']=  df_['TENSION_STAB2'] * df_['COURANT_STAB2']

    for i in range(1,11):
        df_['Ajustage_J'+str(i)] = df_['TENSION_J'+str(i)] * df_['COURANT_J'+str(i)]
    return df_

def calculate_columns_Z_and_S_2(df):
    df_out = df.copy()

    df_out['Z_moy'] = 0
    df_out['S_moy'] = 0
    for i in range(5):
#        df_out['Z_C' + str(i+1)] = df_out['TENSION_C' + str(i+1)] / df_out['COURANT_C' + str(i+1)]
        df_out['Z_C' + str(i+1)] = df_out[['TENSION_C' + str(i+1), 'COURANT_C' + str(i+1)]].apply(calculate_Z, axis = 1)
        df_out['Z_moy'] += df_out['Z_C' + str(i+1)]/5.0
        df_out['S_C' + str(i+1)] = df_out['TENSION_C' + str(i+1)] * df_out['COURANT_C' + str(i+1)]
        df_out['S_moy'] += df_out['S_C' + str(i+1)]/5.0

    for i in range(10):
#        df_out['Z_J' + str(i+1)] = df_out['TENSION_J' + str(i+1)] / df_out['COURANT_J' + str(i+1)]
        df_out['Z_J' + str(i+1)] = df_out[['TENSION_J' + str(i+1), 'COURANT_J' + str(i+1)]].apply(calculate_Z, axis = 1)
        df_out['S_J' + str(i+1)] = df_out['TENSION_J' + str(i+1)] * df_out['COURANT_J' + str(i+1)]
        
    for i in range(2):
#        df_out['Z_STAB' + str(i+1)] = df_out['TENSION_STAB' + str(i+1)] / df_out['COURANT_STAB' + str(i+1)]
        df_out['Z_STAB' + str(i+1)] = df_out[['TENSION_STAB' + str(i+1), 'COURANT_STAB' + str(i+1)]].apply(calculate_Z, axis = 1)
        df_out['S_STAB' + str(i+1)] = df_out['TENSION_STAB' + str(i+1)] * df_out['COURANT_STAB' + str(i+1)]
    
    return df_out
#%% Import and functions
def pipe(folder):
    print('getting data')
    df = get_all_data_in_folder(folder)
    

    print('cleaning  data')
    df_cleaned = clean_df(df)

    print('Calculating Z and S')
    df_filled = calculate_columns_Z_and_S(df_cleaned)

    print('Storing data')
    #df_filled.to_pickle(folder + '.pkl')
    df_filled.to_parquet(folder + '.gzip', engine='pyarrow')
    # save a sample to excel
    lst = [
    'TENSION_C1', 'COURANT_C1', 'Z_C1', 'S_C1', 
    'TENSION_C2', 'COURANT_C2', 'Z_C2', 'S_C2',
    'TENSION_C3', 'COURANT_C3', 'Z_C3', 'S_C3',
    'TENSION_C4', 'COURANT_C4', 'Z_C4', 'S_C4',
    'TENSION_C5', 'COURANT_C5', 'Z_C5', 'S_C5',
    'Z_moy', 'S_moy']

    print('Extracting an excel sample')
    #df_filled.sample(1048)[lst].to_excel(folder + '_sample_3.xlsx')
    df_filled.sample(1048).to_excel(folder + '_sample_3.xlsx')
    return df_filled
def remove_Nan(df,perc):
    
    min_count =  int(((100-perc)/100)*df.shape[0] + 1)
    df_out = df.dropna( axis=1, 
                thresh=min_count)
    return df_out

def generateSummary(df,path):
        # Total missing
        total_miss = df.isnull().sum()
        # Missing precentage
        perc_miss = total_miss/df.isnull().count()*100
        # Type of columns
        col_type = df.dtypes
        # Unique values count
        unique_count = df.nunique()
        # Let's put all of this in a dataframe
        df_summary = pd.DataFrame({'Total missing':total_miss, '% missing': perc_miss, 'Column type': col_type,'unique count': unique_count})
        # Let's sort columns by the percentage of missing values
        df_summary = df_summary.sort_values(by = '% missing',ascending = False)
        # Let's save it into a csv file
        df_summary.T.to_csv(path, encoding='‘utf-8-sig')
        return df_summary
    
#%% Generate Base
if __name__ == '__main__':
    root = r'C:\Users\myedroud\Documents\Project Hager' + '\\'
    root20 = root + r'database\database_piece\input\2020' + '\\'
    root21 = root + r'database\database_piece\input\2021' + '\\'
    root22 = root + r'database\database_piece\input\2022' + '\\'
    subs = [r'2020_01', r'2020_02', r'2020_03', r'2020_04']
    #subs = [r'2020_01']
    frames=[]
    for sub in subs:
        folder = root20 + sub
        frames.append(pipe(folder))
    for sub in subs:
        folder = root21 + sub
        frames.append(pipe(folder))
    for sub in subs:
        folder = root22 + sub
        frames.append(pipe(folder))
        
    result = pd.concat(frames)
    #remove_Nan(result,75)

    result.to_parquet(r'C:\Users\myedroud\Documents\Project Hager\database\database_piece\output\data_par_piece' + '.gzip', engine='pyarrow')
    generateSummary(result,r'C:\Users\myedroud\Documents\Project Hager\database\database_piece\output\summary_data_par_piece.csv')

    cleared_=remove_Nan(result,75)
    cleared_.to_parquet(r'C:\Users\myedroud\Documents\Project Hager\database\database_piece\output\data_par_piece_clean.gzip',engine='pyarrow')
    generateSummary(cleared_,r'C:\Users\myedroud\Documents\Project Hager\database\database_piece\output\summary_data_par_piece_clean.csv')


#%% 
#import pandas as pd
#df = pd.read_pickle(r'C:\Users\jaguiarh\Codes\hager\Project Hager\database\database_piece\2020\2020_01.pkl')
#a=pd.read_parquet(r'C:\Users\myedroud\Documents\Project Hager\database\database_piece\output\data_par_piece.gzip')
#print(a.info())

# %%
import pandas as pd
def remove_Nan(df,perc) -> pd.DataFrame:
    
    min_count =  int(((100-perc)/100)*df.shape[0] + 1)
    df_out = df.dropna( axis=1, 
                thresh=min_count)
    return df_out

def generateSummary(df,path):
        # Total missing
        total_miss = df.isnull().sum()
        # Missing precentage
        perc_miss = total_miss/df.isnull().count()*100
        # Type of columns
        col_type = df.dtypes
        # Unique values count
        unique_count = df.nunique()
        # Let's put all of this in a dataframe
        df_summary = pd.DataFrame({'Total missing':total_miss, '% missing': perc_miss, 'Column type': col_type,'unique count': unique_count})
        # Let's sort columns by the percentage of missing values
        df_summary = df_summary.sort_values(by = '% missing',ascending = False)
        # Let's save it into a csv file
        df_summary.T.to_csv(path, encoding='‘utf-8-sig')
        return df_summary
data=pd.read_parquet(r'C:\Users\myedroud\Documents\Project Hager\database\database_piece\output\data_par_piece.gzip')

cleared_=remove_Nan(data,75)
cleared_.to_parquet(r'C:\Users\myedroud\Documents\Project Hager\database\database_piece\output\data_par_piece_clean.gzip',engine='pyarrow')
generateSummary(cleared_,r'C:\Users\myedroud\Documents\Project Hager\database\database_piece\output\summary_data_par_piece_clean.csv')

# %%
import pandas as pd
#data=pd.read_parquet(r'C:\Users\myedroud\Documents\Project Hager\database\database_piece\output\data_par_piece_clean.gzip')
df=data['NUMERO_LOT'].describe()
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)].reset_index()
prin(df.info())

# %%
