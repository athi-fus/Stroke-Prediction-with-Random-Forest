import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


df = pd.read_csv('healthcare-dataset-stroke-data.csv')
print("cols: {}".format(df.columns))
"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~A~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
print("Dataset-Dataframe Analysis:\n{}".format(df.info()))
print("\n----------------------------------------------------------\n")
print(df)
print("\n----------------------------------------------------------\n")



#----------------using matplotlib-------------------------------------------------------------------------------
gen = df['gender'].value_counts() #frequencies of different values
#print(gen)
mylabels = ["Female", "Male", "Other"]
plt.title("Gender", loc = 'left')
plt.pie(gen, labels = mylabels, autopct='%1.1f%%', startangle=90)
plt.show()

plt.title("Age", loc = 'left')
df["age"].plot(kind = 'hist', bins=20,alpha=0.7, rwidth=0.9, grid=True)
plt.show()

hyp = df['hypertension'].value_counts() #frequencies of different values
#print(hyp)
my_labels = ["No", "Yes"]
plt.title("Hypertension", loc = 'left')
idxhyp = pd.Index(hyp) # Creating the index
plt.bar(my_labels, idxhyp, color = "mediumslateblue")
plt.show()    


heart = df['heart_disease'].value_counts() #frequencies of different values
#print(heart)
plt.title("Heart Disease", loc = 'left')
my_labels=["No", "Yes"]
idxhea = pd.Index(heart) # Creating the index
plt.bar(my_labels, idxhea, color = "#4CAF50")
plt.show()    


wed = df['ever_married'].value_counts() #frequencies of different values
#print(wed)
mylabels = ["Yes", "No"]
plt.title("Ever Married", loc = 'left')
plt.pie(wed, labels = mylabels, autopct='%1.1f%%', startangle=90)
plt.show()

wrk = df['work_type'].value_counts() #frequencies of different values
#print(wrk)
my_labels = ["Private", "Self_Employed", "Children", "Goverment Job", "Never Worked"]
plt.title("Work Types", loc = 'left')
idxwrk = pd.Index(wrk) # Creating the index
plt.bar(my_labels, idxwrk, color = "lightcoral")
plt.show()

res = df['Residence_type'].value_counts() #frequencies of different values
print(res)
mylabels = ["Urban", "Rural"]
plt.title("Residence Types", loc = 'left')
mycolors = ["darkgray", "forestgreen"]
plt.pie(res, labels = mylabels, autopct='%1.1f%%', colors = mycolors, startangle=90)
plt.show()


plt.title("Average Glugose Levels", loc = 'left')
df["avg_glucose_level"].plot(kind = 'hist', bins=20,alpha=0.7, rwidth=0.9, color = "mediumseagreen", grid=True)
plt.show()

plt.title("BMIs", loc = 'left')
df["bmi"].plot(kind = 'hist', bins=20,alpha=0.7, rwidth=0.9, color = "mediumpurple", grid=True)
plt.show()

smoke = df['smoking_status'].value_counts() #frequencies of different values
#print(smoke)
mylabels = ["Never Smoked", "Unknown", "Formerly Smoked", "Smokes"]
plt.title("Smoking Status", loc = 'left')
plt.pie(smoke, labels = mylabels, autopct='%1.1f%%', startangle=90)
plt.show()

strok = df['stroke'].value_counts() #frequencies of different values
#print(strok)
mylabels = ["No", "Yes"]
plt.title("Has had stroke", loc = 'left')
mycolors = ["yellow", "dimgray"]
plt.pie(strok, labels = mylabels, autopct='%1.1f%%', colors = mycolors, startangle=10)
plt.show()

#----------------using seaborn-------------------------------------------------------------------------------
df_arithm =df[['age','avg_glucose_level','bmi','stroke',
               'gender','hypertension','heart_disease',
               'ever_married','work_type','Residence_type']]


g = sns.pairplot(df_arithm, hue = "stroke")
plt.show()


features = ['gender','hypertension','heart_disease',
            'ever_married','work_type','Residence_type']

for f in features:
    plt.figure()
    ax = sns.countplot(x=f, hue="stroke", palette='PuRd_r', data=df_arithm )




"""~~B~~""" #find which columns I should drop
for item in df.columns:
    print(item)
    print("{}".format(df[item].unique()))

count = df['smoking_status'].value_counts() #frequencies of different values
print(count)

genderzz = df['gender'].value_counts() #frequencies of different values
print("genderzzz: {}".format(genderzz))
df.drop(df[df['gender'] == 'Other'].index, inplace = True)
genderzz = df['gender'].value_counts() #frequencies of different values
print("genderzzz: {}".format(genderzz))


"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~B1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
dfB1 = df.drop(['bmi', 'smoking_status' ], axis = 1)
print("dfB1\n{}".format(dfB1.info()))


"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~B2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
dfB2=df.copy()
x = dfB2["bmi"].mean() #calculate the mean of all values of the bmi column
dfB2["bmi"].fillna(x, inplace=True) #fill the missing values with 
print("~dfB2 info~")
print(dfB2.info())
print("\n----------------------------------------------------------\n")

countB = dfB2['smoking_status'].value_counts().idxmax()
#print(countB)
dfB2.loc[(dfB2.smoking_status == 'Unknown'),'smoking_status']= countB #change value 'Unknown' with the value that appears the most in smoking_status


"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~B3~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
dfB3=df.copy()

"""ONE HOT ENCODING THE CATEGORICAL DATA"""
le = preprocessing.LabelEncoder()
dfB3['gender'] = le.fit_transform(dfB3.gender.values) 

dfB3['ever_married'] = le.fit_transform(dfB3.ever_married.values) 

dfB3 = pd.concat([dfB3,pd.get_dummies(dfB3['work_type'], prefix='work')],axis=1) # use pd.concat to join the new columns with your original dataframe
dfB3.drop(['work_type'],axis=1, inplace=True) # now drop the original 'country' column (you don't need it anymore)
dfB3.drop(['work_Never_worked'],axis=1, inplace=True) #drop one of the dummy columns to fight the DUMMY TRAP (dun dun duuuun)

dfB3['Residence_type'] = le.fit_transform(dfB3.Residence_type.values) 

#dfB3.drop(['smoking_status'],axis=1, inplace=True) # now drop the original 'country' column (you don't need it anymore)

print("-------after one hot-------")
print(dfB3.info())
print("\n")

print("\n")

"""-----------------------------F O R   B M I   F I L L - I N-----------------------------"""
"""~~~Splitting the data into training and testing sets~~~"""
dfB3_null_bmi =dfB3[dfB3['bmi'].isnull()] #the NaN rows in the bmi column -- keeping to fill using Linear Regression
dfB3_null_bmi = dfB3_null_bmi.copy()


dfB3_clean_bmi = dfB3.copy() #passing the clean data into a new dataframe
dfB3_clean_bmi.dropna(subset=['bmi'], inplace = True) #dropping the NaN values

print("----------------d f B 3 _ c l e a n----------------")
print(dfB3_clean_bmi)

#cor_matrix = dfB3_clean_bmi.corr() #-----checking the correletion of bmi with the other variables
#print(cor_matrix['bmi'])

"""~~~Choosing the data I'm going to use for training and testing my model~~~"""

x_data = dfB3_clean_bmi[['age', 'hypertension', 'heart_disease',
               'avg_glucose_level', 'stroke', 'gender',
               'ever_married', 'work_Govt_job', 'work_Private',
               'work_Self-employed', 'work_children','Residence_type']]#.values
y_data = dfB3_clean_bmi['bmi']#.values

#x_data = StandardScaler().fit_transform(x_data)
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data ,test_size = 0.25, random_state=111, shuffle=True) #split the data to training and test sets

# fit the linear regression model

lm = LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

## The line / model
plt.scatter(y_test, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()

print ("Linear Regr Score for bmi column: {}\n\n".format(model.score(X_test, y_test)))

#fill-in the missing values
x_data= dfB3_null_bmi[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'stroke',
               'gender', 'ever_married',
               'work_Govt_job', 'work_Private', 'work_Self-employed', 'work_children',
               'Residence_type']]
y_pred = model.predict(x_data)
dfB3_null_bmi['bmi']=y_pred

dfB3_fin_a = pd.concat([dfB3_clean_bmi,dfB3_null_bmi])
print(dfB3_fin_a.info())
print("dfB3_fin_a:\n{}".format(dfB3_fin_a))


"""-----------------------------F O R   s m o k i n g _ s t a t u s   F I L L - I N-----------------------------"""
"""~~~Splitting the data into training and testing sets~~~"""
dfB3_null_smoke =dfB3[(dfB3['smoking_status'] == 'Unknown')] #the NaN rows in the smoking_status column -- keeping to fill using Linear Regression

dfB3_clean_smoke = dfB3.copy() #creating a new dataframe to pass on the clean data

indexNames = dfB3_clean_smoke[dfB3_clean_smoke['smoking_status'] == 'Unknown'].index
dfB3_clean_smoke.drop(indexNames , inplace=True) # Deleting the rows that have the "Unknown" value in the smoking_status column


print("----------------d f B 3 _ c l e a n _ s m o k e----------------")
print(dfB3_clean_smoke['smoking_status'])
print(dfB3_clean_smoke.info())

le = preprocessing.LabelEncoder()
dfB3_clean_smoke['smoking_status'] = le.fit_transform(dfB3_clean_smoke.smoking_status.values) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

print("----------------d f B 3 _ c l e a n _ s m o k e AFTER ENCODING----------------")
print(dfB3_clean_smoke['smoking_status'])
#print(dfB3_clean_smoke.info())

cor_matrix_smoke = dfB3_clean_smoke.corr() #-----checking the correletion of smoking_status with the other variables
print(cor_matrix_smoke['smoking_status'])

#pass the data into the model for training

"""~~~Choosing the data I'm going to use for training and testing my model~~~"""
x_data_s = dfB3_clean_smoke[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'stroke',
               'gender', 'ever_married',
               'work_Govt_job', 'work_Private', 'work_Self-employed', 'work_children',
               'Residence_type']]
y_data_s = dfB3_clean_smoke['smoking_status']

#x_data = StandardScaler().fit_transform(x_data)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(x_data_s, y_data_s ,test_size = 0.25, random_state=111, shuffle=True) #split the data to training and test sets

# fit the linear regression model

model_s = lm.fit(X_train_s, y_train_s)
predictions_s = lm.predict(X_test_s)

## The line / model
plt.scatter(y_test_s, predictions_s)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()

print ("Linear Regr Score for smoking_status column: {}\n\n".format(model_s.score(X_test_s, y_test_s)))

# calculate the new values and fill in the gaps
x_data_s= dfB3_null_smoke[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'stroke',
               'gender', 'ever_married',
               'work_Govt_job',  'work_Private', 'work_Self-employed', 'work_children',
               'Residence_type']]
y_pred_s = model_s.predict(x_data_s)
y_pred_s = [round(x) for x in y_pred_s] #round floats to integers, because we want smoking_status to be a categorical attribute, just in arithmetic form

dfB3_null_smoke['smoking_status']=y_pred_s

dfB3_fin_b = pd.concat([dfB3_clean_smoke,dfB3_null_smoke])
print(dfB3_fin_b.info())




print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


dfB3_final = pd.merge(dfB3_fin_b[['id','smoking_status']], dfB3_fin_a[['id', 'age', 'hypertension',
               'heart_disease', 'avg_glucose_level', 'stroke', 'gender', 'ever_married',
               'work_Govt_job', 'work_Private', 'work_Self-employed', 'work_children',
               'Residence_type','bmi']], on=["id"])


print("Final Dataframe for B3 - filling missing values with linear regression")
print(dfB3_final.info())
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")



"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~B4~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

"""-----------------------------F O R   B M I   F I L L - I N-----------------------------"""
#finding the best parameters for the KNN model and developing the model

parameters = {"n_neighbors": range(1, 50)}
gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
gridsearch.fit(X_train, y_train) #the training set we created in the last subquestion

k=gridsearch.best_params_['n_neighbors']
print("best params: {} neighbors".format(k))

train_preds_grid = gridsearch.predict(X_train) #predicting for the train set
train_mse = mean_squared_error(y_train, train_preds_grid)
train_rmse = sqrt(train_mse)
test_preds_grid = gridsearch.predict(X_test) #predicting for the test set
test_mse = mean_squared_error(y_test, test_preds_grid)
test_rmse = sqrt(test_mse)
print("KNN scores for bmi prediction")
print ("Score KNN: {}".format(gridsearch.score(X_test, y_test)))
print("train_rmse: {}".format(train_rmse))
print("test_rmse: {}".format(test_rmse))

#fill-in the missing values
x_data= dfB3_null_bmi[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'stroke',
               'gender', 'ever_married',
               'work_Govt_job',  'work_Private', 'work_Self-employed', 'work_children',
               'Residence_type']]
y_pred = gridsearch.predict(x_data)
dfB3_null_bmi['bmi']=y_pred
#print(dfB3_null)
dfB4_fin_a = pd.concat([dfB3_clean_bmi,dfB3_null_bmi])
print(dfB4_fin_a.info())
print("dfB4_fin_a:\n{}".format(dfB4_fin_a))


"""-----------------------------F O R   s m o k i n g _ s t a t u s   F I L L - I N-----------------------------"""
#finding the best parameters for the KNN model and developing the model
parameters_s = {"n_neighbors": range(1, 50)}
gridsearch_s = GridSearchCV(KNeighborsRegressor(), parameters_s)
gridsearch_s.fit(X_train_s, y_train_s) #the training set we created in the last subquestion
k_s=gridsearch_s.best_params_['n_neighbors']
print("best params: {} neighbors".format(k_s))


train_preds_grid_s = gridsearch_s.predict(X_train_s) #predicting for the train set
train_mse_s = mean_squared_error(y_train_s, train_preds_grid_s)
train_rmse_s = sqrt(train_mse_s)
test_preds_grid_s = gridsearch_s.predict(X_test_s) #predicting for the test set
test_mse_s = mean_squared_error(y_test_s, test_preds_grid_s)
test_rmse_s = sqrt(test_mse_s)

print("KNN scores for smoking_status prediction")
print ("Score KNN: {}".format(gridsearch_s.score(X_test_s, y_test_s)))
print("train_rmse: {}".format(train_rmse_s))
print("test_rmse: {}".format(test_rmse_s))

#find the missing values with the KNN model and fill them in
#fill-in the missing values
x_data_s= dfB3_null_smoke[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'stroke',
               'gender', 'ever_married',
               'work_Govt_job',  'work_Private', 'work_Self-employed', 'work_children',
               'Residence_type']]
y_pred_s = gridsearch_s.predict(x_data_s)
dfB3_null_smoke['smoking_status']=y_pred_s
#print(dfB3_null)
dfB4_fin_b = pd.concat([dfB3_clean_smoke,dfB3_null_smoke])
print(dfB4_fin_b.info())
print("dfB4_fin_b:\n{}".format(dfB4_fin_b))

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

dfB4_final = pd.merge(dfB4_fin_b[['id','smoking_status']], dfB4_fin_a[['id', 'age', 'hypertension', 
               'heart_disease', 'avg_glucose_level', 'stroke', 'gender', 'ever_married',
               'work_Govt_job', 'work_Private', 'work_Self-employed', 'work_children',
               'Residence_type', 'bmi']], on=["id"])

print("Final Dataframe for B4 - filling missing values with KNN")
print(dfB4_final.info())
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")



"""~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~G~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""
#All my dataframes:
#dfB1
#dfB2
#dfB3_final
#dfB4_final

#df_B1
"""ONE HOT ENCODING THE CATEGORICAL DATA"""
le = preprocessing.LabelEncoder()
dfB1['gender'] = le.fit_transform(dfB1.gender.values) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

dfB1['ever_married'] = le.fit_transform(dfB1.ever_married.values) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

dfB1 = pd.concat([dfB1,pd.get_dummies(dfB1['work_type'], prefix='work')],axis=1) # use pd.concat to join the new columns with your original dataframe
dfB1.drop(['work_type'],axis=1, inplace=True) # now drop the original 'country' column (you don't need it anymore)
dfB1.drop(['work_Never_worked'],axis=1, inplace=True) #drop one of the dummy columns to fight the DUMMY TRAP (dun dun duuuun)

dfB1['Residence_type'] = le.fit_transform(dfB1.Residence_type.values) #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


"""~~~Choosing the data I'm going to use for training and testing my model~~~"""
x_dataB1 = dfB1[['age', 'hypertension', 'heart_disease',
               'avg_glucose_level', 'gender',
               'ever_married', 'work_Govt_job', 'work_Private',
               'work_Self-employed', 'work_children','Residence_type']]#.values


y_dataB1 = dfB1['stroke']#.values

"""~~~Splitting the data to training and testing~~~"""
X_trainB1, X_testB1, y_trainB1, y_testB1 = train_test_split(x_dataB1, y_dataB1 ,test_size = 0.25, random_state=111, shuffle=True) #split the data to training and test sets

#running random forest
rf = RandomForestClassifier(n_estimators=30, criterion='entropy', random_state=1337, class_weight={0: 0.52, 1: 10.5})
rf.fit(X_trainB1, y_trainB1)
prediction_test = rf.predict(X=X_testB1)

# Accuracy on Test
print("\nOn dfB1")
print("Training Accuracy is: ", rf.score(X_trainB1, y_trainB1))
# Accuracy on Train
print("Testing Accuracy is: ", rf.score(X_testB1, y_testB1))
print("f1 score: {}".format(f1_score(y_testB1, prediction_test, average='macro')))
print("Precision: {}".format(precision_score(y_testB1, prediction_test, average="macro")))
print("Recall: {}".format(recall_score(y_testB1, prediction_test, average="macro")))  


#dfB2
"""ONE HOT ENCODING THE CATEGORICAL DATA"""
le = preprocessing.LabelEncoder()
dfB2['gender'] = le.fit_transform(dfB3.gender.values) 

dfB2['ever_married'] = le.fit_transform(dfB2.ever_married.values) 

dfB2 = pd.concat([dfB2,pd.get_dummies(dfB2['work_type'], prefix='work')],axis=1) # use pd.concat to join the new columns with your original dataframe
dfB2.drop(['work_type'],axis=1, inplace=True) # now drop the original 'country' column (you don't need it anymore)
dfB2.drop(['work_Never_worked'],axis=1, inplace=True) #drop one of the dummy columns to fight the DUMMY TRAP (dun dun duuuun)

dfB2['Residence_type'] = le.fit_transform(dfB2.Residence_type.values) 

"""LABEL ENCODING THE smoking_status COLUMN"""
le = preprocessing.LabelEncoder()
dfB2['smoking_status'] = le.fit_transform(dfB2.smoking_status.values) 


"""~~~Choosing the data I'm going to use for training and testing my model~~~"""
x_dataB2 = dfB2[['age', 'hypertension', 'heart_disease',
               'avg_glucose_level', 'gender',
               'ever_married', 'work_Govt_job', 'work_Private',
               'work_Self-employed', 'work_children','Residence_type', 'smoking_status', 'bmi']]#.values
y_dataB2 = dfB2['stroke']#.values

"""~~~Splitting the data to training and testing~~~"""
X_trainB2, X_testB2, y_trainB2, y_testB2 = train_test_split(x_dataB2, y_dataB2 ,test_size = 0.25, random_state=111, shuffle=True) #split the data to training and test sets

#running random forest
rf = RandomForestClassifier(n_estimators=30, criterion='entropy', random_state=1337, class_weight={0: 0.52, 1: 10.5})
rf.fit(X_trainB2, y_trainB2)
prediction_test2 = rf.predict(X=X_testB2)

# Accuracy on Test
print("\nOn dfB2")
print("Training Accuracy is: ", rf.score(X_trainB2, y_trainB2))
# Accuracy on Train
print("Testing Accuracy is: ", rf.score(X_testB2, y_testB2))
print("f1 score: {}".format(f1_score(y_testB2, prediction_test2, average='macro')))
print("Precision: {}".format(precision_score(y_testB2, prediction_test2, average="macro")))
print("Recall: {}".format(recall_score(y_testB2, prediction_test2, average="macro")))  


#dfB3

"""~~~Choosing the data I'm going to use for training and testing my model~~~"""
x_dataB3 = dfB3_final[['age', 'hypertension', 'heart_disease',
               'avg_glucose_level', 'gender',
               'ever_married', 'work_Govt_job', 'work_Private',
               'work_Self-employed', 'work_children','Residence_type','smoking_status', 'bmi']]#.values
y_dataB3 = dfB3_final['stroke']#.values

"""~~~Splitting the data to training and testing~~~"""
X_trainB3, X_testB3, y_trainB3, y_testB3 = train_test_split(x_dataB3, y_dataB3 ,test_size = 0.25, random_state=111, shuffle=True) #split the data to training and test sets

#running random forest
rf = RandomForestClassifier(n_estimators=30, criterion='entropy', random_state=1337, class_weight={0: 0.52, 1: 10.5})
rf.fit(X_trainB3, y_trainB3)
prediction_test3 = rf.predict(X=X_testB3)

# Accuracy on Test
print("\nOn dfB3")
print("Training Accuracy is: ", rf.score(X_trainB3, y_trainB3))
# Accuracy on Train
print("Testing Accuracy is: ", rf.score(X_testB3, y_testB3))
print("f1 score: {}".format(f1_score(y_testB3, prediction_test3, average='macro')))
print("Precision: {}".format(precision_score(y_testB3, prediction_test3, average="macro")))
print("Recall: {}".format(recall_score(y_testB3, prediction_test3, average="macro")))  



#dfB4
"""~~~Choosing the data I'm going to use for training and testing my model~~~"""
x_dataB4 = dfB4_final[['age', 'hypertension', 'heart_disease',
               'avg_glucose_level', 'gender',
               'ever_married', 'work_Govt_job', 'work_Private',
               'work_Self-employed', 'work_children','Residence_type', 'smoking_status', 'bmi']]#.values
y_dataB4 = dfB4_final['stroke']#.values

"""~~~Splitting the data to training and testing~~~"""
X_trainB4, X_testB4, y_trainB4, y_testB4 = train_test_split(x_dataB4, y_dataB4 ,test_size = 0.25, random_state=111, shuffle=True) #split the data to training and test sets

#running random forest
rf = RandomForestClassifier(n_estimators=30, criterion='entropy', random_state=1337, class_weight={0: 0.52, 1: 10.5})
rf.fit(X_trainB4, y_trainB4)
prediction_test4 = rf.predict(X=X_testB4)

# Accuracy on Test
print("\nOn dfB4")
print("Training Accuracy is: ", rf.score(X_trainB4, y_trainB4))
# Accuracy on Train
print("Testing Accuracy is: ", rf.score(X_testB4, y_testB4))
print("f1 score: {}".format(f1_score(y_testB4, prediction_test4, average='macro')))
print("Precision: {}".format(precision_score(y_testB4, prediction_test4, average="macro")))
print("Recall: {}".format(recall_score(y_testB4, prediction_test4, average="macro")))  



print("\n\nTuning the Random Forest model")
# Tunning Random Forest 
from itertools import product
n_estimators = 50
max_features = [1, 'sqrt', 'log2']
max_depths = [None, 2, 3, 4, 5, 6, 7]
for f, d in product(max_features, max_depths): # with product we can iterate through all possible combinations
    rf = RandomForestClassifier(n_estimators=n_estimators, 
                                criterion='entropy', 
                                max_features=f, 
                                max_depth=d, 
                                n_jobs=2, class_weight={0: 0.52, 1: 10.5},
                                random_state=1337)
    rf.fit(X_trainB2, y_trainB2)
    prediction_test2 = rf.predict(X=X_testB2)
    prediction_test2 = [round(x) for x in prediction_test2]

    print('Classification accuracy on test set with max features = {} and max_depth = {}: {:.3f}'.format(f, d, accuracy_score(y_testB2,prediction_test2)))
    print("\tf1 score: {}".format(f1_score(y_testB2, prediction_test2, average='macro')))
    print("\tPrecision: {}".format(precision_score(y_testB2, prediction_test2, average='macro', labels=np.unique(prediction_test2))))
    print("\tRecall: {}\n\n".format(recall_score(y_testB2, prediction_test2, average="macro")))  

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
