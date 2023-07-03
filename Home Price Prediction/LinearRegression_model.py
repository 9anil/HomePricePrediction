# Step1- Collecting/Gathering the data
import numpy as np
import pandas as pd
data=pd.read_csv("housing.csv")
#print(data.head())
# Step2- Data cleaning/data Preprocessing/data wrangling- Some categorical data convert
data['mainroad']=data['mainroad'].apply({'no':0,'yes':1}.get)
data['guestroom']=data['guestroom'].apply({'no':0,'yes':1}.get)
data['basement']=data['basement'].apply({'no':0,'yes':1}.get)
data['hotwaterheating']=data['hotwaterheating'].apply({'no':0,'yes':1}.get)
data['airconditioning']=data['airconditioning'].apply({'no':0,'yes':1}.get)
data['prefarea']=data['prefarea'].apply({'no':0,'yes':1}.get)
data['furnishingstatus']=data['furnishingstatus'].apply({'unfurnished':0,'furnished':1,'semi-furnished':2}.get)
#print(data.head())
# Step3- Divide the data into independent(x) and dependent(y) dataset
x=data[['area','bedrooms','bathrooms','stories','mainroad','guestroom','basement','hotwaterheating','airconditioning','parking','prefarea','furnishingstatus']]
y=data[['price']]
# print(x.head())
# print(y.head())
# Step4- Split the data into training and testing set by help of sklearn library
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)# test_size=0.2 means 20% data use for testing & remaining use for learning
# Creating- Linear regression model
from sklearn.linear_model import LinearRegression
regression=LinearRegression() # LinearRegression model activated hear
regression.fit(x_train,y_train)# fit() use to train the mechine learning model
print(regression.predict(x_test))#To predict the price of houses from LinearRegression model
print("Accuracy score:",regression.score(x,y))
# To predict price of house of inserting the inputs
new_house_data={'area':1000,'bedrooms':2,'bathrooms':1,'stories':2,'mainroad':1,'guestroom':1,'basement':0,'hotwaterheating':1,'airconditioning':1,'parking':0,'prefarea':0,'furnishingstatus':1}
new_house=pd.DataFrame(new_house_data,index=[1])
#print(new_house)
print(regression.predict(new_house))
