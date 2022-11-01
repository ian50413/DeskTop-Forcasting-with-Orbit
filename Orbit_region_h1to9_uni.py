#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 10:28:31 2022

@author: ian
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 14:31:25 2022

@author: ian

Region
"""

#%%extract roadmap as single
%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import orbit
from orbit.models import LGT, DLT
from orbit.diagnostics.backtest import BackTester, TimeSeriesSplitter
from orbit.diagnostics.plot import plot_bt_predictions
from orbit.diagnostics.metrics import smape, wmape
from orbit.utils.dataset import load_iclaims
from orbit.diagnostics.plot import plot_predicted_data
import warnings
warnings.filterwarnings('ignore')

# roadmap_df_ori=pd.read_csv('MONTHLY_DT_ROADMAP_FOB_WITH_2ARIMA_LAG.csv', index_col=0)
roadmap_df_ori=pd.read_csv('MONTHLY_DT_ROADMAP_FOB.csv', index_col=0)
roadmap_df_ori = roadmap_df_ori[['time_period','region','roadmap_group','roadmap_invoice_qty']]
                                 # ,'roadmap_invoice_qty_lag_1','roadmap_invoice_qty_lag_12']]

dict_region={}
for region in roadmap_df_ori.region.unique():
    dict_region[region]=pd.DataFrame()
for (region, roadmap_group), roadmap_df in roadmap_df_ori.groupby(by=['region','roadmap_group']):
    rr=roadmap_df.copy()
    dict_region[f'{region}']=dict_region[f'{region}'].append(rr, ignore_index=True)
    # df=dict_region[f'{region}']
    """
    Change Region Below
    """
    #Type GC TWN CHINA below
    df=dict_region['CHINA']


alpha={}
for hrz in range(1,10):
    # df = pd.read_csv('MONTHLY_DT_WW_ROADMAP_FOB_clean.csv')
    # df.dropna(inplace=True)
    # df.rename(columns = {'Unnamed: 0': 'count'}, inplace = True)

    #change timestamp format
    df['time_period'] = pd.DatetimeIndex(df['time_period'])
    dct = {}
    for idx, v in enumerate(df['roadmap_group'].unique()):
        dct[f'df{idx}'] = df.loc[df['roadmap_group'] == v]
    # print(dct['df0'])
    fcst_output = pd.DataFrame()
    # hori=9
    roadmap=idx+1#full roadmap = idx+1
    beta=pd.DataFrame()
    
    for i in range(roadmap):# for i in range(idx+1):
    # for _, roadmap_df in df.groupby(by=['region','roadmap_group'])
        #plot single roadmap 
        # fig, axs = plt.subplots(2, 2,figsize=(20,8))
        # axs[0, 0].plot(dct[f'df{i}']['time_period'], dct[f'df{i}']['roadmap_invoice_qty'])
        # axs[0, 0].set_title('roadmap_invoice_qty')
        # axs[0, 1].plot(dct[f'df{i}']['time_period'], dct[f'df{i}']['roadmap_t1ci_qty'], 'tab:orange')
        # axs[0, 1].set_title('roadmap_t1ci_qty')
        # axs[1, 0].plot(dct[f'df{i}']['time_period'], dct[f'df{i}']['roadmap_t2ci_qty'], 'tab:green')
        # axs[1, 0].set_title('roadmap_t2ci_qty')        
        
        #check roadmap data row count and whether fulfill horizon 9 condition
        print('Processing horizon '+str(hrz)+ ' currently.')
        print(dct[f'df{i}'].roadmap_group.values[0]+ ' has '\
              +str(len(dct[f'df{i}'].index))+' rows.')
            
        if len(dct[f'df{i}'].index) >= 21:
            print(dct[f'df{i}'].roadmap_group.values[0]+' fulfill horizon 9 span.')
        else:
            print(dct[f'df{i}'].roadmap_group.values[0]+' has only '+ str(len(dct[f'df{i}'].index)-12)+' more rows.')
            
        #The min limit of train set should > 12(response to test_size)
        # try:
        dlt = DLT(
            date_col='time_period',
            response_col='roadmap_invoice_qty',
            # regressor_col=['roadmap_t1ci_qty', 'roadmap_t2ci_qty'],
            seasonality=12,
            estimator='stan-map',
        )
        
        # configs
        # if len(dct[f'df{i}'].roadmap_group.index)-12 >= 9:
        try:
            min_train_len = 12
            forecast_len = hrz
            incremental_len = 1
            window_type = 'expanding'
            bt = BackTester(
                model=dlt,
                df=dct[f'df{i}'],
                min_train_len=min_train_len,
                incremental_len=incremental_len,
                forecast_len=forecast_len,
                window_type=window_type,
            )
            bt.fit_predict()
            
            #below code r some coef ,mape ,plot
            # # print(bt.score(metrics=[mse_naive, naive_error]))
            # fitted_models = bt.get_fitted_models()
            # model_1 = fitted_models[0]
            # # print(model_1.get_regression_coefs())
            predicted_df = bt.get_predicted_df()
            # print(predicted_df.head())
            
        #print seperate train test plot
            # ts_splitter = bt.get_splitter()
            # _ = ts_splitter.plot()
        except:
            pass
    
        #get each horizon with last
        key=list(predicted_df.split_key.unique())
        target_list=[]
        for t in key:
            target_row=predicted_df.loc[predicted_df['split_key'] == t, 'training_data']
            target_row=target_row.replace(True, 1)
            target_row=target_row.replace(False, 0)
            target_row=target_row.to_frame(name='slpit_key')
            target_list.append(target_row.index[target_row['slpit_key'] == 0].values[-1])
        
        #extract predicted row data with assigned horizon
        h = pd.DataFrame()
        for target in target_list:
            result = predicted_df.iloc[[target]]
            h=h.append(result, ignore_index=True)
        # h[f'horizon_{hori}'] = h[f'horizon_{hori}'].filter(items=['date', 'prediction'])
        
        dct[f'df{i}'] = pd.merge(dct[f'df{i}'], h,  how='left', left_on=['time_period'], right_on = ['date'])
        dct[f'df{i}'] = dct[f'df{i}'].filter(items=['time_period', 'roadmap_group', 'prediction'])
        
        #need change name base on diff h and add a column name horizon with refered value then append 1-9
        fcst_output = fcst_output.append(dct[f'df{i}'], ignore_index=True)
        
    #Create new column for each horizon        
    alpha[f'alpha_{hrz}']=fcst_output.copy()
    alpha[f'alpha_{hrz}']['Horizon'] = f'{hrz}'
#Concate all horizon df into dict alpha
for alpha_all in range(9):
    beta = beta.append(alpha[f'alpha_{alpha_all+1}'], ignore_index=True)

"""
Change Region Below and Save file as region name
"""
beta['Region'] = 'CHINA'
beta.to_csv('Orbit_full_h_CHINA.csv')
#%%Create new column for each horizon
# alpha={}
# beta=pd.DataFrame()
alpha[f'alpha_{hrz}']=fcst_output.copy()
alpha[f'alpha_{hrz}']['Horizon'] = f'{hrz}'

#%%Concate all horizon df into dict alpha
for alpha_all in range(9):
    print(alpha_all)
    beta = beta.append(alpha[f'alpha_{alpha_all+1}'], ignore_index=True)

beta.to_csv('The_Ultimate_Horizon_DT_FCST_with_Orbit.csv')

    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        