# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 17:50:54 2026

@author: HP
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df=pd.read_csv("C:\Assignments\Tumor_Ensemble.csv")

df.dtypes
print("Shape of dataset:", df.shape)
print("Size of dataset:", df.size)
#Shape of dataset: (569, 32)
#Size of dataset: 18208
#2Second Moment: Variance/Standard Deviation (Dispersion)
var_values = df.var(numeric_only=True)
std_values = df.std(numeric_only=True)
print("\nVariance (Second Moment):\n", var_values)
print("\nStandard Deviation (Second Moment):\n", std_values)
#The most of the data is variaced

#3 third Moment: skewness
skew_values=df.skew(numeric_only=True)
print("\nSkewness (Third Moment):\n", skew_values)
#Fourth Moment: Kurtosis
kurt_values=df.kurtosis(numeric_only=True)
print("\nKurtosis (Fourth Moment):\n", kurt_values)
#Histogram

df.drop(columns=['id']).hist(
    figsize=(12,10),
    edgecolor='black'
)
plt.suptitle("Histograms of Tumor_Ensemble")
plt.tight_layout()
plt.show()

#central tendencies
print("\nMedian:\n", df.median(numeric_only=True))
print("\nMode:\n", df.mode(numeric_only=True))

#Scatter Plot: 
sns.scatterplot(data=df, x='id', y='diagnosis', hue='dimension_worst')
plt.title("Scatter Plot: id vs diagnosis")
plt.show()
#Joint Plot: 
sns.jointplot(data=df, x="id", y="diagnosis", kind="scatter")
plt.show()
# Pair Plot
sns.pairplot(df, hue="diagnosis")
plt.subtitle("Pair Plot of All Features", y=1.02)
plt.show()
#Boxplot: Detect outliers
plt.figure(figsize=(10,6))
sns.boxplot(data=df.drop(columns=['diagnosis']), orient='h')
plt.title('Boxplot of All Features')
plt.show()
df.dtypes
#Heatmap (Correlation)
plt.figure(figsize=(10,8))
sns.heatmap(
    df.select_dtypes(include=['float64']).corr(),
    annot=True,
    cmap='coolwarm')
plt.title("Correlation Heatmap - Company Dataset")
plt.show()

#PDF & CDF ANALYSIS
num_cols=df.select_dtypes(include=np.number).columns.drop(['radius_mean'])
for col in num_cols:
    plt.figure(figsize=(12,5))
    #pdf
    plt.subplot(1,2,1)
    sns.kdeplot(df[col], fill=True)
    plt.title(f"PDF of {col}")
    #CDF
    plt.subplot(1,2,2)
    sorted_vals=np.sort(df[col])
    y_vals=np.arange(len(sorted_vals))/float(len(sorted_vals))
    plt.plot(sorted_vals, y_vals, marker='.', linestyle='none')
    plt.title(f"CDF of {col}")
    plt.tight_layout()
    plt.show()

#CLASS DISTRIBUTION (TARGET ANALYSIS)
###############
sns.countplot(x=df['radius_mean'])
plt.title("Distribution of radius_mean")
plt.show()
####################################
#DATA PREPROCESSING
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#
#step2: Load the Dataset
#
df=pd.read_csv("C:\Assignments\Tumor_Ensemble.csv") 
print("Initial Shape", df.shape)
df.head()
#
#Step3: Basic Data Quality Check
df.info()
print("\nMissing Values:\n", df.isnull().sum())
#inference
#-Dataset has no missing values
#-All input feature numeric
#-Target column 'Type' is integer encoded
#
#
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["diagnosis"] = le.fit_transform(df["diagnosis"])

#Step4: Outlier Detection (From EDA)
#
cols=[['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'points_mean', 'symmetry_mean', 'dimension_mean', 'radius_se',
       'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'points_se', 'symmetry_se',
       'dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
       'area_worst', 'smoothness_worst', 'compactness_worst',
       'concavity_worst', 'points_worst', 'symmetry_worst', 'dimension_worst']]

plt.figure(figsize=(12,6))
sns.boxplot(data=df.drop(columns=['diagnosis']), orient='h')
plt.title("Boxplot of Tumor_Ensemble Features")
plt.show()

#
#Step5: Outlier Treatment Using Winsorization
#
def winsorize_column(df, col):
    winsor = Winsorizer(
        capping_method='iqr',
        tail='both',
        fold=1.5,
        variables=[col]
    )
    return winsor.fit_transform(df[[col]])
for col in ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'points_mean', 'symmetry_mean', 'dimension_mean', 'radius_se',
       'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'points_se', 'symmetry_se',
       'dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
       'area_worst', 'smoothness_worst', 'compactness_worst',
       'concavity_worst', 'points_worst', 'symmetry_worst', 'dimension_worst']:
    df[col]=winsorize_column(df, col)
print("Outlier treatment completed")
#
#Step7: Target variable Label Encoding
#
#Convert numeric class labels into meaningful

df['diagnosis']=np.where(df['diagnosis']==1, 'built_win_fl', df['diagnosis'])
df['diagnosis']=np.where(df['diagnosis']==2, 'built_win_nfl', df['diagnosis'])
df['diagnosis']=np.where(df['diagnosis']==3, 'veh_win_fl', df['diagnosis'])
df['diagnosis']=np.where(df['diagnosis']==4, 'veh_win_nfl', df['diagnosis'])
df['diagnosis']=np.where(df['diagnosis']==5, 'containers', df['diagnosis'])
df['diagnosis']=np.where(df['diagnosis']==6, 'tableware', df['diagnosis'])
df['diagnosis']=np.where(df['diagnosis']==7, 'headlamps', df['diagnosis'])

df['diagnosis'].value_counts()
#
#Step8: Feature Scaling - MIN-MAX NORMALIZATION
#
x_features=df.drop(columns=['diagnosis'])
scaler=MinMaxScaler()
x_scaled=scaler.fit_transform(x_features)
x_scaled=pd.DataFrame(x_scaled, columns=x_features.columns)
x_scaled.describe()
#
#Step9: SPLIT INPUT AND OUTPUT
#
x=np.array(x_scaled)
y=np.array(df['diagnosis'])
#
#Step10: TRAIN-TEST-SPLIT
#
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)
print("Training set size:", x_train.shape)
print("Testing set size", x_test.shape) 
#
#FINAL PREPROCESSING SUMMARY
#
#Identifier removed
# Outlier treated using Winsorization
# Target labels made interpretable
# Feature 

print("Tumor_Ensemble Dataset Preprocessing completed sucessfully!")


#######################################################


import pandas as pd
from sklearn.ensemble import StackingClassifier as SKStacking
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn import model_selection
from sklearn import datasets

df=pd.read_csv("C:\Assignments\Tumor_Ensemble.csv") 
print(df.head())
print(df.columns)
X = df.drop(columns=['id', 'diagnosis'])
y = df['diagnosis']
print(X.shape)
print(y.shape)
estimators=[
    ('knn', KNeighborsClassifier(n_neighbors=1)),
    ('rf', RandomForestClassifier(random_state=1)),
    ('gnb', GaussianNB())
]
stack = SKStacking(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=500),
    cv=3
)
scores=model_selection.cross_val_score(stack, X, y, cv=3, scoring='accuracy')
print("Stacking (sklearn) CV accuracy:", scores.mean())
