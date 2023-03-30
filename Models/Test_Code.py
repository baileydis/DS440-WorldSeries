import numpy as py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from flaml import AutoML
import tabula as tb
import re
from sklearn.model_selection import train_test_split



##load data##
pitching_data = pd.read_csv('teampitching.csv')
batting_data = pd.read_csv('teambattingstats.csv')
playoffs_data = pd.read_csv('playoffappearances.csv')
master_ws_data = pd.read_csv('masterWS.csv', header=0)

###CLEAN DATA###

#create predictor variable Win_Percent as percent wins in the season
master_ws_data["Win_Percent"] = master_ws_data["W"] / (master_ws_data["W"] + master_ws_data["L"])
#Remove the Wins/Loses columns and case identifier variables Season and Team
master_ws_data = master_ws_data.drop(columns = ["Season","Team",'W', 'L'])
#Cast strings to floats and remove JSON char
master_ws_data= master_ws_data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
master_ws_data = master_ws_data.replace('%', '', regex=True)
master_ws_data = master_ws_data.astype(float)
print(master_ws_data.head())

lr = LinearRegression()


# Seperate predictor and response variables
X = master_ws_data.iloc[:,:-1].astype(float)
Y = master_ws_data["Win_Percent"]

print(X)
print(Y)

#Split data randomly into test and train
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=12)

#Fit linear regression using training data and test for accuracy using train
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("R2 score:", r2) #R^2 Metric for accuracy

automl = AutoML()


#Auto ML Settings
automl_settings = {
    "time_budget": 60, #seconds
    "metric": 'r2',
    "task": 'regression'
    }


#automl.fit(X_train.values, y_train, **automl_setting,)
# Predict
print(automl.predict(X_train).shape)
# Export the best model
print(automl.model)

#correlation heatmap moving to Data Visualization

import seaborn as sns

# calculate correlation matrix
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(20,20))

# plot heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

#Correlation with output variable
cor_target = abs(corr_matrix["W"])
#Selecting highly correlated features
relevant_features_W = cor_target[cor_target>0.5]
relevant_features_W

#Correlation with output variable
cor_target = abs(corr_matrix["L"])
#Selecting highly correlated features
relevant_features_L = cor_target[cor_target>0.5]
relevant_features_L

#I have it saved to my google drive right now. Biggest correlation for W/L is Strikes, Balls, Pitches, SO, BB, RBI, R, HR, X3B, X2B< X1B, H, PA, AB.




