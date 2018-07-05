#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 13:13:50 2018

@author: shobanamahadevan
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import functools as ft
import glob as glob

""""
os.chdir("/Users/shobanamahadevan/Documents/Mona_ML_prac /My_trials/DATASET")
results = pd.DataFrame([])
for counter, file in enumerate(glob.glob("*.csv")):
    namedf = pd.read_csv(file)
    results = results.append(namedf)
 
results.to_csv('/Users/shobanamahadevan/Documents/Mona_ML_prac /My_trials/DATASET/combinedfile.csv')
results = results.dropna()
import glob
import pandas as pd

# get data file names
path =r'/Users/shobanamahadevan/Documents/Mona_ML_prac /My_trials/DATASET'
filenames = pd.read_csv("Qualification-Table 1.csv","Occupation-Table 1.csv","Braodband-Table 1.csv","Location-Table 1.csv")
dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))
# Concatenate all data into one DataFrame
big_frame = pd.concat(dfs, ignore_index=True)
"""

# Importing the datasets
Metro = pd.read_csv('Metro-Table 1.csv')
Qualification = pd.read_csv('Qualification-Table 1.csv')
Credit_Score = pd.read_csv('Credit Score-Table 2.csv')
Occupation = pd.read_csv('Occupation-Table 1.csv')
Braodband = pd.read_csv('Braodband-Table 1.csv')
Location = pd.read_csv('Location-Table 1.csv')
Mobile = pd.read_csv('Mobile Number-Table 1.csv')
"""
comb = ft.reduce(lambda left,right:pd.merge(left,right,on='Mobile Number',how='outer'),
                [Qualification, Occupation, Braodband, Location, Mobile])

metro2 = pd.DataFrame(np.where((Metro['Metro?']=='Yes'), (Metro['Pin code'],Metro['Metro?']),np.nan))
metro2 = metro2.transpose()
metro2 = metro2.dropna()
"""
temp = Location['Pin code']
for x in range (0, len(temp)):
    #if (loc.isin(metro2[0])):
    if(temp[x]==635120 or temp[x]==696253 or temp[x]==629028):
        temp[x] = 'Metro'     
    else:
        temp[x] = 'Non Metro'
Location['Pin code'] = temp

##MATCH THE MOBILE NUMBERS (with old dataset) AND COPY THEIR SCORES 
#merge multiple dataframes together after reduction-reduce(function, sequence[, initial])
comb = ft.reduce(lambda left,right:pd.merge(left,right,on='Mobile Number',how='outer'),
                [Qualification, Occupation, Braodband, Location, Mobile])

#getting the complete un-processed dataset
comb_df = comb.merge(Credit_Score[['Mobile Number','Average Score']],how='inner')

#HANDLING THE  MISSING VALUES
#Delete the missing values
comb_df = comb_df.dropna()

#Splitting into dependent and independent variables
X = comb_df.iloc[:, :-1].values
y1 = comb_df.iloc[:, -1:].values
X = pd.DataFrame(X)
#normalising the score
# normaliseed value = (((current value-old min)*(new max-new min))/(old max-old min)+new min
y = (((y1-300)*(100-0))/(900-300))

#Categorical values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X = X.apply(LabelEncoder().fit_transform)

onehotencoder = OneHotEncoder(categorical_features=[1,2])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding dummy-variable trap
X = np.delete(X, [2,3], axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#fitting the multiple regression model
from  sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predict the test results
y_pred = regressor.predict(X_test)

#x-axis for plotting
z = np.array(range(0,119))


#checking the variance
print('Coefficients: \n', regressor.coef_
# The mean squared error
print("Mean squared error: %.2f" % np.mean((y_pred - y_test) ** 2))
#Calculating variance score to get accuracy
# Explained variance score: 1 is perfect prediction
print('Variance score: %.0f' % regressor.score(X_test, y_test))


# Visualising the Polynomial Regression results
#plt.scatter((y_pred-y_test), y_pred, color = 'red')
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
plt.figure()
#plt.boxplot(y_pred,1)
#plt.scatter(z,y_test,color="green")
plt.plot(z, y_pred, color = 'magenta',label = 'y_pred')
plt.plot(z,y_test,color='blue', label = 'y_test')
plt.title('difference')
plt.xlabel('z')
plt.ylabel('y')
plt.legend()
plt.show()


"""
filenames = glob.glob("*.csv")
list_of_dfs = [pd.read_csv(filename) for filename in filenames]
for dataframe, filename in zip(list_of_dfs, filenames)
  dataframe['filename'] = filename
combined_df = pd.concat(list_of_dfs, ignore_index=True)
frame = pd.DataFrame()
dataframes = [ ]
for f in filenames:
    data = pd.read_csv(f)
    df = dataframes.append(data)
frame = pd.concat(dataframes)


df[df.name != 'Tina']
Metro['Metro?'] = Metro['Metro?' != 'No']

if (Metro['Metro?'].any()=='No'):
    Metro.drop(['No'])
    #Metro['Metro?']=='Nan'
else:
    Metro['Metro?']=='Yes'
    

Location2 = Location['Mobile Number'].merge(Location[Pin code'].isin(metro2[0]))

Location2 = Location
Location2.loc[Location2['Pin code']==635120,'Pin code'] ='Metro'
Location2.loc[Location2['Pin code']==696253,'Pin code'] ='Metro'
Location2.loc[Location2['Pin code']==629028,'Pin code'] ='Metro'
    

labelencoder_X = LabelEncoder()
X[:,1] = labelencoder_X.fit_transform(X[:,1])
X[:,2] = labelencoder_X.fit_transform(X[:,2])
X[:,5] = labelencoder_X.fit_transform(X[:,5])
X[:,6] = labelencoder_X.fit_transform(X[:,6])
X[:,8] = labelencoder_X.fit_transform(X[:,8])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""
"""



