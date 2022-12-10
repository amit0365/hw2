import pandas as pd
import numpy as np

all_stocks=pd.read_csv("sp500_stocks.csv")
all_stocks['Symbol'].unique()

#likely more than 500 because of index changes
len(all_stocks['Symbol'].unique())

sector=pd.read_csv("sp500_companies.csv")

sector

all_stocks=all_stocks.merge(sector[['Symbol','Sector']],how='left',on='Symbol')
all_stocks

all_stocks['Sector'].unique()

#we sill have some missing sectors - figure out what to do here
all_stocks[all_stocks['Sector'].isnull()]['Symbol'].unique()

all_stocks[all_stocks['Symbol']=='GOOG']

# all_stocks.loc[all_stocks['Sector'].isnull(),'Sector']='UNKNOWN'

# YOU MAY NEED TO UPDATE THIS LOGIC
all_stocks.loc[all_stocks['Symbol']=='CEG','Sector']='Utilities'
all_stocks.loc[all_stocks['Symbol']=='ELV','Sector']='Healthcare'
all_stocks.loc[all_stocks['Symbol']=='META','Sector']='Communication Services'
all_stocks.loc[all_stocks['Symbol']=='PARA','Sector']='Communication Services'
all_stocks.loc[all_stocks['Symbol']=='SBUX','Sector']='Consumer Cyclical'
all_stocks.loc[all_stocks['Symbol']=='V','Sector']='Financial Services'
all_stocks.loc[all_stocks['Symbol']=='WBD','Sector']='Communication Services'
all_stocks.loc[all_stocks['Symbol']=='WTW','Sector']='Financial Services'
all_stocks.loc[all_stocks['Sector'].isnull(),'Sector']='UNKNOWN'

all_stocks['Sector'].unique()

"""### In order to understand what factors may be driving returns we first need to calcualte returns"""

#calculate return as a log-difference
all_stocks=all_stocks.sort_values(['Symbol','Date']).reset_index(drop=True)
all_stocks['adj_close_lag1']=all_stocks[['Symbol','Date','Adj Close']].groupby(['Symbol']).shift(1)['Adj Close'].reset_index(drop=True)
all_stocks['return']=np.log(all_stocks['Adj Close']/all_stocks['adj_close_lag1'])

"""### Think about how to use the other features - DO NOT USE FEATURES FROM TODAY TO MODEL TODAY'S RETURN"""

all_stocks

def create_lagged_features(df,var):
    df[var+'_lag1']=df[['Symbol','Date',var]].groupby(['Symbol']).shift(1)[var].reset_index(drop=True)
    df[var+'_lag2']=df[['Symbol','Date',var]].groupby(['Symbol']).shift(2)[var].reset_index(drop=True)
    df[var+'_rolling5']=df[['Symbol','Date',var+'_lag1']].groupby(['Symbol'])[var+'_lag1'].rolling(5).sum().reset_index(drop=True)
    df[var+'_rolling15']=df[['Symbol','Date',var+'_lag1']].groupby(['Symbol'])[var+'_lag1'].rolling(15).sum().reset_index(drop=True)
    return df

all_stocks=create_lagged_features(all_stocks,'return')
all_stocks

all_stocks=create_lagged_features(all_stocks,'Volume')
all_stocks

all_stocks['relative_vol_1_15']=all_stocks['Volume_lag1']/all_stocks['Volume_rolling15']
all_stocks['relative_vol_5_15']=all_stocks['Volume_rolling5']/all_stocks['Volume_rolling15']

all_stocks=create_lagged_features(all_stocks,'Open')
all_stocks.head(10)

all_stocks.columns

"""### Transform Sector for modeling"""

sector_counts=all_stocks['Sector'].value_counts()
sector_counts

#perform frequency based encoding (usually this would only use training porttion to fit transform, but need to keep transform constant across days)
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder(categories=[list(sector_counts.index)])

all_stocks['Sector_enc']=enc.fit_transform(all_stocks[['Sector']])

"""### Let's pick a stock to see what might be driving returns for that stock based on modeling the market"""

import shap
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split

this_stock='WMT'
feature_list=['Sector_enc','return_lag1','return_lag2','return_rolling5','return_rolling15','relative_vol_1_15','relative_vol_5_15','Open_lag1','Open_lag2', 'Open_rolling5',
       'Open_rolling15']

last10days=[]

for i in range(3,13):
  last10days.append(all_stocks.loc[all_stocks.index[-i],'Date'])
print(last10days)

"""## DAY -1"""

feature_list=['Sector_enc','return_lag1','return_lag2','return_rolling5','return_rolling15','relative_vol_1_15','relative_vol_5_15','Open_lag1','Open_lag2', 'Open_rolling5',
       'Open_rolling15']
this_date=last10days[0]
print(this_date)

#create a list of today's stocks EXCLUDING the one we are interested in
today_stocks=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']!=this_stock)]

#create a train/test split for early stopping. Just using about 50 stocks
X_train, X_test, y_train, y_test = train_test_split(today_stocks[feature_list], today_stocks['return'], test_size=0.1, random_state=42)

param_grid = {'max_depth':list(range(3,7,1))}
params_fit={"eval_metric" : "mae",'eval_set': [[X_test, y_test]],'early_stopping_rounds':10}

gbm = xgb.XGBRegressor(colsample_bylevel=1, colsample_bynode=1, colsample_bytree=.75, gamma=0,learning_rate=0.05, max_delta_step=0,
             missing=-99999, n_estimators=300, random_state=0,reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=.5, verbosity=1)

search = GridSearchCV(gbm,param_grid=param_grid, verbose=1)
search.fit(X_train,y_train,**params_fit)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

#best model was max depth 3 and we have that estimator fit on whole training data (excluding the early stopping piece)
search.best_estimator_

search.best_estimator_.feature_importances_

#input for only this stock
this_data=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']==this_stock)][feature_list]
this_actual=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']==this_stock)]['return']
search.best_estimator_.predict(this_data), this_actual

explainer = shap.TreeExplainer(search.best_estimator_)
shap_values = explainer.shap_values(this_data)

pip install streamlit

# visualize the prediction's explanation
shap.initjs()
plot=[]
plot.append(shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0]))
plot[0]

#the baseline value
explainer.expected_value

shap_values

"""## DAY -2 GD

"""

this_date=last10days[1]
print(this_date)

#create a list of today's stocks EXCLUDING the one we are interested in
today_stocks=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']!=this_stock)]

#create a train/test split for early stopping. Just using about 50 stocks
X_train, X_test, y_train, y_test = train_test_split(today_stocks[feature_list], today_stocks['return'], test_size=0.1, random_state=42)

param_grid = {'max_depth':list(range(3,7,1))}
params_fit={"eval_metric" : "mae",'eval_set': [[X_test, y_test]],'early_stopping_rounds':10}

gbm = xgb.XGBRegressor(colsample_bylevel=1, colsample_bynode=1, colsample_bytree=.75, gamma=0,learning_rate=0.05, max_delta_step=0,
             missing=-99999, n_estimators=300, random_state=0,reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=.5, verbosity=1)

search = GridSearchCV(gbm,param_grid=param_grid, verbose=1)
search.fit(X_train,y_train,**params_fit)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

#best model was max depth 3 and we have that estimator fit on whole training data (excluding the early stopping piece)
search.best_estimator_

search.best_estimator_.feature_importances_

#input for only this stock
this_data=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']==this_stock)][feature_list]
this_actual=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']==this_stock)]['return']
search.best_estimator_.predict(this_data), this_actual

explainer = shap.TreeExplainer(search.best_estimator_)
shap_values = explainer.shap_values(this_data)

# visualize the prediction's explanation
shap.initjs()
plot.append(shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:]))
plot[1]

#the baseline value
explainer.expected_value

shap_values

"""## DAY -3"""

this_date=last10days[2]
print(this_date)

#create a list of today's stocks EXCLUDING the one we are interested in
today_stocks=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']!=this_stock)]

#create a train/test split for early stopping. Just using about 50 stocks
X_train, X_test, y_train, y_test = train_test_split(today_stocks[feature_list], today_stocks['return'], test_size=0.1, random_state=42)

param_grid = {'max_depth':list(range(3,7,1))}
params_fit={"eval_metric" : "mae",'eval_set': [[X_test, y_test]],'early_stopping_rounds':10}

gbm = xgb.XGBRegressor(colsample_bylevel=1, colsample_bynode=1, colsample_bytree=.75, gamma=0,learning_rate=0.05, max_delta_step=0,
             missing=-99999, n_estimators=300, random_state=0,reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=.5, verbosity=1)

search = GridSearchCV(gbm,param_grid=param_grid, verbose=1)
search.fit(X_train,y_train,**params_fit)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

#best model was max depth 3 and we have that estimator fit on whole training data (excluding the early stopping piece)
search.best_estimator_

search.best_estimator_.feature_importances_

#input for only this stock
this_data=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']==this_stock)][feature_list]
this_actual=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']==this_stock)]['return']
search.best_estimator_.predict(this_data), this_actual

explainer = shap.TreeExplainer(search.best_estimator_)
shap_values = explainer.shap_values(this_data)

# visualize the prediction's explanation
shap.initjs()
plot3=shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])
plot3

#the baseline value
explainer.expected_value

shap_values

"""## DAY -4"""

this_date=last10days[3]
print(this_date)

#create a list of today's stocks EXCLUDING the one we are interested in
today_stocks=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']!=this_stock)]

#create a train/test split for early stopping. Just using about 50 stocks
X_train, X_test, y_train, y_test = train_test_split(today_stocks[feature_list], today_stocks['return'], test_size=0.1, random_state=42)

param_grid = {'max_depth':list(range(3,7,1))}
params_fit={"eval_metric" : "mae",'eval_set': [[X_test, y_test]],'early_stopping_rounds':10}

gbm = xgb.XGBRegressor(colsample_bylevel=1, colsample_bynode=1, colsample_bytree=.75, gamma=0,learning_rate=0.05, max_delta_step=0,
             missing=-99999, n_estimators=300, random_state=0,reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=.5, verbosity=1)

search = GridSearchCV(gbm,param_grid=param_grid, verbose=1)
search.fit(X_train,y_train,**params_fit)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

#best model was max depth 3 and we have that estimator fit on whole training data (excluding the early stopping piece)
search.best_estimator_

search.best_estimator_.feature_importances_

#input for only this stock
this_data=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']==this_stock)][feature_list]
this_actual=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']==this_stock)]['return']
search.best_estimator_.predict(this_data), this_actual

explainer = shap.TreeExplainer(search.best_estimator_)
shap_values = explainer.shap_values(this_data)

# visualize the prediction's explanation
shap.initjs()
plot4=shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])
plot4

#the baseline value
explainer.expected_value

shap_values

"""## DAY -5 GD"""

this_date=last10days[-4]
print(this_date)

#create a list of today's stocks EXCLUDING the one we are interested in
today_stocks=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']!=this_stock)]

#create a train/test split for early stopping. Just using about 50 stocks
X_train, X_test, y_train, y_test = train_test_split(today_stocks[feature_list], today_stocks['return'], test_size=0.1, random_state=42)

param_grid = {'max_depth':list(range(3,7,1))}
params_fit={"eval_metric" : "mae",'eval_set': [[X_test, y_test]],'early_stopping_rounds':10}

gbm = xgb.XGBRegressor(colsample_bylevel=1, colsample_bynode=1, colsample_bytree=.75, gamma=0,learning_rate=0.05, max_delta_step=0,
             missing=-99999, n_estimators=300, random_state=0,reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=.5, verbosity=1)

search = GridSearchCV(gbm,param_grid=param_grid, verbose=1)
search.fit(X_train,y_train,**params_fit)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

#best model was max depth 3 and we have that estimator fit on whole training data (excluding the early stopping piece)
search.best_estimator_

search.best_estimator_.feature_importances_

#input for only this stock
this_data=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']==this_stock)][feature_list]
this_actual=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']==this_stock)]['return']
search.best_estimator_.predict(this_data), this_actual

explainer = shap.TreeExplainer(search.best_estimator_)
shap_values = explainer.shap_values(this_data)

# visualize the prediction's explanation
shap.initjs()
plot5=shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])
plot5

#the baseline value
explainer.expected_value

shap_values

"""## DAY -6"""

this_date=last10days[5]
print(this_date)

#create a list of today's stocks EXCLUDING the one we are interested in
today_stocks=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']!=this_stock)]

#create a train/test split for early stopping. Just using about 50 stocks
X_train, X_test, y_train, y_test = train_test_split(today_stocks[feature_list], today_stocks['return'], test_size=0.1, random_state=42)

param_grid = {'max_depth':list(range(3,7,1))}
params_fit={"eval_metric" : "mae",'eval_set': [[X_test, y_test]],'early_stopping_rounds':10}

gbm = xgb.XGBRegressor(colsample_bylevel=1, colsample_bynode=1, colsample_bytree=.75, gamma=0,learning_rate=0.05, max_delta_step=0,
             missing=-99999, n_estimators=300, random_state=0,reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=.5, verbosity=1)

search = GridSearchCV(gbm,param_grid=param_grid, verbose=1)
search.fit(X_train,y_train,**params_fit)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

#best model was max depth 3 and we have that estimator fit on whole training data (excluding the early stopping piece)
search.best_estimator_

search.best_estimator_.feature_importances_

#input for only this stock
this_data=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']==this_stock)][feature_list]
this_actual=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']==this_stock)]['return']
search.best_estimator_.predict(this_data), this_actual

explainer = shap.TreeExplainer(search.best_estimator_)
shap_values = explainer.shap_values(this_data)

# visualize the prediction's explanation
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])

#the baseline value
explainer.expected_value

shap_values

"""## DAY -7 GD"""

this_date=last10days[6]
print(this_date)

#create a list of today's stocks EXCLUDING the one we are interested in
today_stocks=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']!=this_stock)]

#create a train/test split for early stopping. Just using about 50 stocks
X_train, X_test, y_train, y_test = train_test_split(today_stocks[feature_list], today_stocks['return'], test_size=0.1, random_state=42)

param_grid = {'max_depth':list(range(3,7,1))}
params_fit={"eval_metric" : "mae",'eval_set': [[X_test, y_test]],'early_stopping_rounds':10}

gbm = xgb.XGBRegressor(colsample_bylevel=1, colsample_bynode=1, colsample_bytree=.75, gamma=0,learning_rate=0.05, max_delta_step=0,
             missing=-99999, n_estimators=300, random_state=0,reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=.5, verbosity=1)

search = GridSearchCV(gbm,param_grid=param_grid, verbose=1)
search.fit(X_train,y_train,**params_fit)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

#best model was max depth 3 and we have that estimator fit on whole training data (excluding the early stopping piece)
search.best_estimator_

search.best_estimator_.feature_importances_

#input for only this stock
this_data=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']==this_stock)][feature_list]
this_actual=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']==this_stock)]['return']
search.best_estimator_.predict(this_data), this_actual

explainer = shap.TreeExplainer(search.best_estimator_)
shap_values = explainer.shap_values(this_data)

# visualize the prediction's explanation
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])

#the baseline value
explainer.expected_value

shap_values

"""## DAY -8"""

this_date=last10days[7]
print(this_date)

#create a list of today's stocks EXCLUDING the one we are interested in
today_stocks=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']!=this_stock)]

#create a train/test split for early stopping. Just using about 50 stocks
X_train, X_test, y_train, y_test = train_test_split(today_stocks[feature_list], today_stocks['return'], test_size=0.1, random_state=42)

param_grid = {'max_depth':list(range(3,7,1))}
params_fit={"eval_metric" : "mae",'eval_set': [[X_test, y_test]],'early_stopping_rounds':10}

gbm = xgb.XGBRegressor(colsample_bylevel=1, colsample_bynode=1, colsample_bytree=.75, gamma=0,learning_rate=0.05, max_delta_step=0,
             missing=-99999, n_estimators=300, random_state=0,reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=.5, verbosity=1)

search = GridSearchCV(gbm,param_grid=param_grid, verbose=1)
search.fit(X_train,y_train,**params_fit)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

#best model was max depth 3 and we have that estimator fit on whole training data (excluding the early stopping piece)
search.best_estimator_

search.best_estimator_.feature_importances_

#input for only this stock
this_data=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']==this_stock)][feature_list]
this_actual=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']==this_stock)]['return']
search.best_estimator_.predict(this_data), this_actual

explainer = shap.TreeExplainer(search.best_estimator_)
shap_values = explainer.shap_values(this_data)

# visualize the prediction's explanation
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])

#the baseline value
explainer.expected_value

shap_values

"""## DAY -9 GD"""

this_date=last10days[8]
print(this_date)

#create a list of today's stocks EXCLUDING the one we are interested in
today_stocks=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']!=this_stock)]

#create a train/test split for early stopping. Just using about 50 stocks
X_train, X_test, y_train, y_test = train_test_split(today_stocks[feature_list], today_stocks['return'], test_size=0.1, random_state=42)

param_grid = {'max_depth':list(range(3,7,1))}
params_fit={"eval_metric" : "mae",'eval_set': [[X_test, y_test]],'early_stopping_rounds':10}

gbm = xgb.XGBRegressor(colsample_bylevel=1, colsample_bynode=1, colsample_bytree=.75, gamma=0,learning_rate=0.05, max_delta_step=0,
             missing=-99999, n_estimators=300, random_state=0,reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=.5, verbosity=1)

search = GridSearchCV(gbm,param_grid=param_grid, verbose=1)
search.fit(X_train,y_train,**params_fit)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

#best model was max depth 3 and we have that estimator fit on whole training data (excluding the early stopping piece)
search.best_estimator_

search.best_estimator_.feature_importances_

#input for only this stock
this_data=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']==this_stock)][feature_list]
this_actual=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']==this_stock)]['return']
search.best_estimator_.predict(this_data), this_actual

explainer = shap.TreeExplainer(search.best_estimator_)
shap_values = explainer.shap_values(this_data)

# visualize the prediction's explanation
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])

#the baseline value
explainer.expected_value

shap_values

"""# VIZ"""

shap.initjs()
plot[0]

"""# For the assignment you will 
* design features 
* create a model each day for the last 10 trading days 
* attribute returns each day to your features
* create a stacked bar chart over time attributing returns to SHAP base,feature attributions, and model error for your selected stock
* create docker image that creates this visualiztion on a website (flask, streamlit, plotly dash or other)
"""
