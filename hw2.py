import pandas as pd
import numpy as np

all_stocks=pd.read_csv("sp500_stocks.csv");

sector=pd.read_csv("sp500_companies.csv");

all_stocks=all_stocks.merge(sector[['Symbol','Sector']],how='left',on='Symbol');


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



#calculate return as a log-difference
all_stocks=all_stocks.sort_values(['Symbol','Date']).reset_index(drop=True);
all_stocks['adj_close_lag1']=all_stocks[['Symbol','Date','Adj Close']].groupby(['Symbol']).shift(1)['Adj Close'].reset_index(drop=True);
all_stocks['return']=np.log(all_stocks['Adj Close']/all_stocks['adj_close_lag1']);


def create_lagged_features(df,var):
    df[var+'_lag1']=df[['Symbol','Date',var]].groupby(['Symbol']).shift(1)[var].reset_index(drop=True)
    df[var+'_lag2']=df[['Symbol','Date',var]].groupby(['Symbol']).shift(2)[var].reset_index(drop=True)
    df[var+'_rolling5']=df[['Symbol','Date',var+'_lag1']].groupby(['Symbol'])[var+'_lag1'].rolling(5).sum().reset_index(drop=True)
    df[var+'_rolling15']=df[['Symbol','Date',var+'_lag1']].groupby(['Symbol'])[var+'_lag1'].rolling(15).sum().reset_index(drop=True)
    return df

all_stocks=create_lagged_features(all_stocks,'return')

all_stocks=create_lagged_features(all_stocks,'Volume')

all_stocks['relative_vol_1_15']=all_stocks['Volume_lag1']/all_stocks['Volume_rolling15']
all_stocks['relative_vol_5_15']=all_stocks['Volume_rolling5']/all_stocks['Volume_rolling15']

all_stocks=create_lagged_features(all_stocks,'Open')

sector_counts=all_stocks['Sector'].value_counts()

#perform frequency based encoding (usually this would only use training porttion to fit transform, but need to keep transform constant across days)
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder(categories=[list(sector_counts.index)])

all_stocks['Sector_enc']=enc.fit_transform(all_stocks[['Sector']])

import shap
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split

this_stock='WMT'
feature_list=['Sector_enc','return_lag1','return_lag2','return_rolling5','return_rolling15','relative_vol_1_15','relative_vol_5_15','Open_lag1','Open_lag2', 'Open_rolling5',
       'Open_rolling15']

last10days=[]

for i in range(3,13):
  last10days.append(all_stocks.loc[all_stocks.index[-i],'Date'])


"""## DAY -1"""

feature_list=['Sector_enc','return_lag1','return_lag2','return_rolling5','return_rolling15','relative_vol_1_15','relative_vol_5_15','Open_lag1','Open_lag2', 'Open_rolling5',
       'Open_rolling15']
this_date=last10days[0]
print(this_date,this_stock)

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
search.fit(X_train,y_train,**params_fit);

#input for only this stock
this_data=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']==this_stock)][feature_list]
this_actual=all_stocks[np.logical_and(all_stocks['Date']==this_date,all_stocks['Symbol']==this_stock)]['return']
search.best_estimator_.predict(this_data), this_actual;



# visualize the prediction's explanation

import shap
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

@st.cache
def load_data():
    return shap.datasets.boston()

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

st.title("SHAP in Streamlit")

explainer = shap.TreeExplainer(search.best_estimator_)
shap_values = explainer.shap_values(this_data)


# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0]))
#st.bar_chart(chart_data)


#the baseline value
explainer.expected_value;

shap_values;

