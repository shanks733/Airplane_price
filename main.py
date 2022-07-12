import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def timeconv(x):
    x.replace
    hr = ''
    min = ''
    hc = False
    for i in range(len(x)-1):
        if x[i] == 'h':
            hc = True
            i+=1
        if hc == False:
            hr += x[i]
        elif hc == True:
            min += x[i]
    if min == '':
        time =int(hr)*60
    else :
        time = int(hr)*60 + int(min)
    return time


data = pd.read_excel('Data_Train.xlsx',sheet_name='Sheet1')
data.drop(['Route','Dep_Time','Arrival_Time'],axis=1,inplace = True)
data.dropna(inplace = True)
print(data)
print(data.info())
data['Duration'] = data['Duration'].apply(timeconv)
data['Total_Stops']=data['Total_Stops'].apply(lambda x:0 if x == 'non-stop' else x[0])
print(data)
print(data['Total_Stops'].value_counts())
print(data['Additional_Info'].value_counts())
print(data.head())
print(data['Duration'])
data['day'] = pd.DatetimeIndex(data['Date_of_Journey']).day
data['year'] = pd.DatetimeIndex(data['Date_of_Journey']).year
data['month'] = pd.DatetimeIndex(data['Date_of_Journey']).month
data['weekday'] = pd.DatetimeIndex(data['Date_of_Journey']).weekday
data.drop(['Date_of_Journey','weekday'],axis=1,inplace = True)
data = pd.get_dummies(data,columns=['Airline','Source','Destination','Additional_Info'],drop_first=True)
sns.heatmap(data.corr())
print(data.head(10).to_string())
X = data.loc[:,data.columns != 'Price']
y = data['Price']
print(X)
print(y)
X['Total_Stops'] = X['Total_Stops'].astype('int')
print(X.dtypes)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print(X.head(10).to_string())
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[['Duration','month']] = sc.fit_transform(X_train[['Duration','month']])
X_test[['Duration','month']] = sc.transform(X_test[['Duration','month']])

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

from sklearn.metrics import r2_score
train_scr = r2_score(y_train,y_pred_train)
test_scr = r2_score(y_test,y_pred_test)
print(train_scr,"       ",test_scr)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train,y_train)
y_pred_train2 = regressor.predict(X_train)
y_pred_test2 = regressor.predict(X_test)
print("train ",r2_score(y_train, y_pred_train2))
print("test",r2_score(y_test, y_pred_test2))
print('='*40)
from sklearn.model_selection import GridSearchCV
param_dist = {
    "criterion" : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'max_depth' : range(2,60),
    'max_features' : ['auto', 'sqrt', 'log2'],
    'min_samples_split': range(2,20)
}
gridmodel = GridSearchCV(regressor,param_grid=param_dist,n_jobs=-1,cv=5)
gridmodel.fit(X_train,y_train)
print(gridmodel.best_estimator_)