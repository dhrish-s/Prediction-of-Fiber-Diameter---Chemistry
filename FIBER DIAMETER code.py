import pandas as pd
import numpy as np
import streamlit as st
#import matplotlib.pyplot as plt
#import tensorflow as tf
#from tensorflow import keras
from sklearn.model_selection import train_test_split
#from keras.models import Sequential
#from keras.layers import Dense
#from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

st.write("## Lab Data for fiber diameter")
df=pd.read_excel("fiber_data.xlsx")
st.dataframe(df, use_container_width=True)
df['Source'] = df['Source'].fillna(method='ffill')

#df=df.iloc[:,[0,5,6,7,8,9,10]]
df=df.iloc[:,[5,6,7,8,9,10,11]]

x=df.iloc[:,:-1]

x=x.replace('-',0)

x['Voltage (kV)'] = x['Voltage (kV)'].astype(float)
x['Distance (cm)'] = x['Distance (cm)'].astype(float)
x['Feed (mL/h)'] = x['Feed (mL/h)'].astype(float)

y=df.iloc[:,-1]
y=y.replace('-',0)


mm=MinMaxScaler()
x_norm=mm.fit_transform(x)

X_train,X_test,Y_train,Y_test= train_test_split(x,y,test_size=0.2,random_state=0)



lr=LinearRegression()

#lasso=Lasso(alpha=0.1)
#poly=PolynomialFeatures(degree=2)
#x_poly=poly.fit_transform(x_norm)
#X_train,X_test,Y_train,Y_test= train_test_split(x,y,test_size=0.2,random_state=0)

lr.fit(X_train,Y_train)
#lasso.fit(X_train,Y_train)
y_predict=lr.predict(X_test)


score=r2_score(Y_test,y_predict)

fiber = pd.concat([x, y], axis=1)

#DecisionTree
decision_tree=DecisionTreeRegressor(max_depth=3)
decision_tree.fit(X_train,Y_train)

dt_pred=decision_tree.predict(X_test)
accuracy = r2_score(Y_test, dt_pred)

#GradientBooster


gbm = GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=0)
gbm.fit(X_train, Y_train)

gbm_pred = gbm.predict(X_test)

gbm_acc = r2_score(Y_test, gbm_pred)
print(gbm_acc)

poly = st.number_input('Polymer Concentration (wt%) 8–30.3', 3.00, 32.00, 15.00)
red = st.number_input('RED 0.0751–0.49', 0.0600, 0.4000, 0.3544)
volt = st.number_input('Voltage (kV) 5-70', 0.0, 80.0, 13.0)
dist = st.number_input('Distance (cm) 3 – 27.7', 3.0, 30.0, 15.0)
feed = st.number_input('Feed(mL/h) 0.06 – 14.21', 0.00, 16.00, 14.21)
flory_x = st.number_input('Flory-Huggins X parameter 0.004 – 0.1651', 0.0040, 0.3000, 0.0887)
new_data=[poly,red,volt,dist,feed,flory_x]
def predict_data(x):
    x=(np.array(x)).reshape(1,6)
    gbm_pred = gbm.predict(x)
    st.write("## Predicted Fiber Diameter:","\n ##",*gbm_pred.tolist(),"nm")
    print(gbm_pred)
predict_data(new_data)

