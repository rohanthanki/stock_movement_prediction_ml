#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries and Creating a Logger

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


# # Custom Functions

# ## Wrangling Data

def wrangle_data(df, ticker):
    df.columns= df.columns.str.title()
    df['Date'] = df['Date'].astype('datetime64[ns]')
    df_w = df.        reset_index().        sort_values(by='Date').        ffill().        bfill()
    df_w['Ticker'] = ticker
    
    return df_w

# ## Generating Features

def create_features(df, vix, rf, ticker):
    
    # Wrangling data
    df_w = wrangle_data(df, ticker)
    
    # Daily stock returns
    df_w['RET_CC'] = df_w['Close'].pct_change(periods=1)

    # Output - Intra-day stock movement 
    df_w['Y'] = (df_w['RET_CC']>0).astype(int)

    # Percentage Volume Change
    df_w['VOLUME_PCT_CHG'] = df_w['Volume'].pct_change(periods=1)

    # Percentage Price Oscialltor
    df_w['MA_5'] = df_w['Close'].rolling(5).mean()
    df_w['MA_22'] = df_w['Close'].rolling(22).mean()
    df_w['MA_66'] = df_w['Close'].rolling(66).mean()
    df_w['PP_OSCILLATOR'] = (df_w['MA_5'] - df_w['MA_22']) / df_w['MA_66']

    # Relative strength indicator
    df_w['RSI'] = RSIIndicator(df_w['Close']).rsi()

    # Bollinger Band Indicators
    indicator_bb = BollingerBands(close=df_w["Close"], window=20, window_dev=2)
    df_w['B_H'] = indicator_bb.bollinger_hband()
    df_w['B_L'] = indicator_bb.bollinger_lband()
    df_w['B_MA'] = indicator_bb.bollinger_mavg()
    df_w['BOLLINGER_HDIFF_PCT'] = (df_w['B_H'] - df_w['B_MA']) / df_w['B_MA']
    df_w['BOLLINGER_LDIFF_PCT'] = (df_w['B_L'] - df_w['B_MA']) / df_w['B_MA']
    
    # Percentage Change in VIX
    df_w = df_w.merge(right=vix, on=['Date'], how='inner').        sort_values(by=['Date'])
    
    # Change in Risk Free Rate
    df_w = df_w.merge(right = rf, on=['Date'], how='inner').        sort_values(by=['Date'])
    
    # Lagging Features by 1 day to prevent look-ahead bias
    df_w['Date'] = df_w['Date'].shift(periods=-1)
    df_w['Y'] = df_w['Y'].shift(periods=-1)   

    # Removing NaN and INF values
    df_w = df_w.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
    
    return df_w

def select_features(df):    
    
    # Selecting the feature matrix
    X = df[['VOLUME_PCT_CHG', 'PP_OSCILLATOR', 'VIX_PCT_CHG', 'RF_CHG', 'BOLLINGER_HDIFF_PCT', 'BOLLINGER_LDIFF_PCT']].to_numpy()
    y = df['Y'].to_numpy()
    
    # Standardising the input features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Returning X and y values
    return(X_scaled,y)


# ## Training and Testing Models

# ### Model 1 - Support Vector Machine (SVM)

def train_test_SVM(X_train, X_test, y_train, y_test, models_summary_SVM, ticker):
    
    # Set SVM parameters for grid search
    svm_tuned_parameters = {
        "kernel": ["rbf", "linear"], 
         "C": [0.25, 0.5, 1, 4, 16, 64, 256, 1024]
    }
    
    # Training SVM
    clf_SVM = GridSearchCV(SVC(), svm_tuned_parameters, scoring='f1', cv=5, n_jobs=-1, verbose=0)
    clf_SVM.fit(X_train, y_train)
    
    # Running the optimally tuned model on the test set to get predicted values
    y_test_pred_SVM = clf_SVM.predict(X_test)
    
    # Adding best parameters and performance metrics to a list and returning them    
    models_summary_SVM['Ticker'].append(ticker)
    
    models_summary_SVM['Best_Params_SVM'].append(clf_SVM.best_params_)    
    models_summary_SVM['Accuracy_SVM'].append(accuracy_score(y_test, y_test_pred_SVM))
    models_summary_SVM['Precision_SVM'].append(precision_score(y_test, y_test_pred_SVM))
    models_summary_SVM['Recall_SVM'].append(recall_score(y_test, y_test_pred_SVM))
    models_summary_SVM['F1_Score_SVM'].append(f1_score(y_test, y_test_pred_SVM))
    models_summary_SVM['AUC_ROC_SVM'].append(roc_auc_score(y_test, y_test_pred_SVM))
    
    return(models_summary_SVM)


# ### Model 2 - K Nearest Neighbors (KNN)

def train_test_KNN(X_train, X_test, y_train, y_test, models_summary_KNN, ticker):
    
    # Set KNN parameters for grid search
    knn_tuned_parameters = {
        'n_neighbors': np.arange(5, 24, 2),
    }
    
    # Training KNN
    clf_KNN = GridSearchCV(KNeighborsClassifier(), knn_tuned_parameters, scoring='f1', cv=5, n_jobs=-1, verbose=0)
    clf_KNN.fit(X_train, y_train)
    
    # Running the optimally tuned model on the test set to get predicted values
    y_test_pred_KNN = clf_KNN.predict(X_test)
    
    # Adding best parameters and performance metrics to a list and returning them    
    models_summary_KNN['Ticker'].append(ticker)
    
    models_summary_KNN['Best_Params_KNN'].append(clf_KNN.best_params_)    
    models_summary_KNN['Accuracy_KNN'].append(accuracy_score(y_test, y_test_pred_KNN))
    models_summary_KNN['Precision_KNN'].append(precision_score(y_test, y_test_pred_KNN))
    models_summary_KNN['Recall_KNN'].append(recall_score(y_test, y_test_pred_KNN))
    models_summary_KNN['F1_Score_KNN'].append(f1_score(y_test, y_test_pred_KNN))
    models_summary_KNN['AUC_ROC_KNN'].append(roc_auc_score(y_test, y_test_pred_KNN))
    
    return(models_summary_KNN)