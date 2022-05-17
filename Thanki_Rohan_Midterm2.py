#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries and Creating a Logger

# In[1]:


import time
start_time = time.time()


# In[2]:

# general python libraries
import os
import glob
import logging

# libraries to extract financial data and compute technical indicators
import yfinance as yf
import pandas_ta as pta
from ta import add_all_ta_features
from ta.utils import dropna
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
from ta.momentum import PercentagePriceOscillator
from ta.trend import macd

# analytical libraries
import numpy as np
import pandas as pd

# ML pre-processing libraries
from sklearn.preprocessing import StandardScaler

# ML hyper-parameter tuning libraries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# ML model libraries
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ML model evaluation
from sklearn.metrics import *

# plotting libraries
import altair as alt

# user defined functions
from Thanki_Rohan_module1 import *

# Creating a logger
logging.basicConfig(filename="Output_Files/Midterm_Project_2.log",
                            filemode='w',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)


# # Running the Main Script

# ## Initialising Global Variables

# ### Declaring Global Variables for part 1

tickers_list = pd.read_csv("tickers.csv")['Ticker'].to_list()
start_date = '2000-01-01'
end_date = '2021-11-12'

models_summary_SVM = {'Ticker':[],
                      'Best_Params_SVM':[], 
                      'Accuracy_SVM':[],
                      'Precision_SVM':[],
                      'Recall_SVM':[],
                      'F1_Score_SVM':[], 
                      'AUC_ROC_SVM':[],
                     }

models_summary_KNN = {'Ticker':[],
                      'Best_Params_KNN':[], 
                      'Accuracy_KNN':[],
                      'Precision_KNN':[],
                      'Recall_KNN':[],
                      'F1_Score_KNN':[], 
                      'AUC_ROC_KNN':[]
                     }


# ### Declaring Global Variables for part 2


path = "stock_dfs"
snp500_data_files = glob.glob(os.path.join(path, "*.csv"))
models_summary_part_2 = {'Ticker':[],
                         'Best_Params_SVM':[], 
                         'Accuracy_SVM':[],
                         'Precision_SVM':[],
                         'Recall_SVM':[],
                         'F1_Score_SVM':[],
                         'AUC_ROC_SVM':[],
                        }


# ### Downloading VIX and Risk Free Rate Data

# Daily Percentage Change in VIX
vix = yf.download(['^VIX'], start=start_date, end=end_date)['Adj Close'].    reset_index().    rename(columns={'Adj Close':'VIX'}).    ffill()
vix['VIX_PCT_CHG'] = vix['VIX'].pct_change()
vix = vix.drop(columns = ['VIX'])


# Interest Rate Change
rf = yf.download(['^TNX'], start=start_date, end=end_date)['Adj Close'].    reset_index().    rename(columns={'Adj Close':'RF'}).    ffill()
rf['RF_CHG'] = rf['RF'].diff()
rf = rf.drop(columns = ['RF'])


# ## Part 1 - 10 Chosen Tickers


for ticker in tickers_list:
    
    try:
    
        # Downloading Data from Yahoo Finance
        data = yf.download(ticker, start=start_date, end=end_date).            reset_index()

        # Create features
        data_w = create_features(data, vix, rf, ticker)

        # Creating input matrix and output vector
        X, y = select_features(data_w)

        # Creating training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

        # Training and testing the models
        models_summary_SVM = train_test_SVM(X_train, X_test, y_train, y_test, models_summary_SVM, ticker)
        models_summary_KNN = train_test_KNN(X_train, X_test, y_train, y_test, models_summary_KNN, ticker)

        logging.info("Ticker " + ticker + " completed")
    
    except Exception as e:
        logging.info("Ticker " + ticker + " is not completed")
        logging.error(e)

# Saving Summary of Best Models
models_summary_part_1 = pd.DataFrame(models_summary_SVM).    merge(pd.DataFrame(models_summary_KNN), on='Ticker', how='inner')
models_summary_part_1.to_csv('Output_Files/Model_Summary_Part_1.csv')
logging.info("Part 1 completed - 10 stocks")


# ## Part 2 - S&P 500 Stocks


path = "stock_dfs"
snp500_data_files = glob.glob(os.path.join(path, "*.csv"))


for file_path in snp500_data_files:
    
    try:
        
        # Extracting the ticker
        ticker = os.path.split(file_path)[-1].split('.')[0]

        # Reading the data
        data = pd.read_csv(file_path)

        # Create features
        data_w = create_features(data, vix, rf, ticker)

        # Creating input matrix and output vector
        X, y = select_features(data_w)

        # Creating training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

        # Training and testing the model that performed the best previously
        models_summary_part_2 = train_test_SVM(X_train, X_test, y_train, y_test, models_summary_part_2, ticker)
    
        logging.info("Ticker " + ticker + " completed")
    
    except Exception as e:
        
        logging.warning("Ticker " + ticker + " is not completed")
        logging.error(e)


# Saving Summary of Best Models
models_summary_part_2_df = pd.DataFrame(models_summary_part_2)
models_summary_part_2_df.to_csv('Output_Files/Model_Summary_Part_2.csv')
logging.info("Part 2 completed - S%P500 stocks")


logging.info("Time taken to complete the program: %s seconds" % (time.time() - start_time))

