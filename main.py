import pandas as pd
import numpy as np
df= pd.read_csv('insurance.csv')
#print(df.head())
from sklearn.preprocessing import LabelEncoder
lb= LabelEncoder()
df['sex']= lb.fit_transform(df['sex'])
df['smoker']= lb.fit_transform(df['smoker'])
df['region']= lb.fit_transform(df['region'])

#print(df.head(3))

x= df.drop(columns=['charges'])
y= df['charges']    

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
lr= LinearRegression()  
lr.fit(x_train, y_train)
y_pred= lr.predict(x_test)
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))