import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import sklearn
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = [10,5]
warnings.simplefilter(action="ignore",category=FutureWarning)

st.set_page_config(page_title="Gradient Boosting Model For Data Analysis",layout="centered",initial_sidebar_state="auto",menu_items=None)
st.title("ANALYZE YOUR DATA FOR TAXIS")
st.header("--Gradient Boosting--")

# dt=sns.load_dataset('taxis')
dt=sns.load_dataset("taxis")
st.write(dt.head(5))
st.write(dt.describe() )                 
# st.write(dt.corr())

dt.drop(["pickup","dropoff","pickup_zone","dropoff_zone","pickup_borough","dropoff_borough"],axis=1,inplace=True)

label={"yellow":1,"red":2,"blue":3,"green":4}
dt["color"]=dt["color"].map(label)

label={"credit card":1,"cash":0}
dt["payment"]=dt["payment"].map(label)


dt.interpolate(method='pad',inplace=True)
# there are no null values in this dataset now.
sns.heatmap(dt.isnull(),yticklabels=False,cbar=False,cmap='tab20c_r')
plt.title("Missing Data:Training Set")
st.write(plt.show())

x=dt.drop("fare",axis=1)
y=dt["fare"]

from sklearn import preprocessing

prepro = preprocessing.StandardScaler().fit(x)

x_trans = prepro.fit_transform(x)

y = y.to_numpy()
y = y.reshape(-1,1)

prepro2= preprocessing.StandardScaler().fit(y)

y_trans = prepro2.fit_transform(y)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_trans,y_trans,test_size=0.30,random_state=101)


from sklearn.ensemble import HistGradientBoostingRegressor

hgb = HistGradientBoostingRegressor()

hgb.fit(x_train, y_train)


y_pred = hgb.predict(x_test)

from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(y_test,y_pred))
# print("The root mean squared error is :",rmse)
st.write(f"The root mean squared error is: {rmse:.2f}")


st.header("--Random Forest--")
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()

rfr.fit(x_train, y_train)

y_pred = rfr.predict(x_test)

rmse=sqrt(mean_squared_error(y_test,y_pred))
# print("The root mean squared error is :",rmse)
st.write(f"The root mean squared error is: {rmse:.2f}")


st.header("--Decision Tree--")
from sklearn.tree import DecisionTreeRegressor

dtr = RandomForestRegressor()

dtr.fit(x_train, y_train)

y_pred = dtr.predict(x_test)

rmse=sqrt(mean_squared_error(y_test,y_pred))
# print("The root mean squared error is :",rmse)
st.write(f"The root mean squared error is: {rmse:.2f}")


st.header("--Linear Regression--")
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

rmse=sqrt(mean_squared_error(y_test,y_pred))
# print("The root mean squared error is :",rmse)
st.write(f"The root mean squared error is: {rmse:.2f}")
