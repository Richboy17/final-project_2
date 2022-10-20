import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.pylabtools import figsize

import folium
from folium.plugins import HeatMap
import plotly.express as px
import seaborn as sns

## Import LabelEncoder from sklearn
from sklearn.preprocessing import LabelEncoder
#Calculate by using PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

df = pd.read_csv('/content/winequality-red.csv', header=0, delimiter=';')

fig = px.histogram(df, x="quality")

fig.show()

dcopy = df.copy()

dcopy.duplicated().value_counts()
dcopy.isnull().sum()

dcopy_new = dcopy.drop_duplicates(inplace=False)
dcopy_new

dcopy_new.plot(kind='box')

red_circle = dict(markerfacecolor='red', marker='o', markeredgecolor='white')

fig, axs = plt.subplots(1, len(dcopy_new.columns), figsize=(40,10))

for i, ax in enumerate(axs.flat):
    ax.boxplot(df.iloc[:,i], flierprops=red_circle)
    ax.set_title(dcopy_new.columns[i], fontsize=20, fontweight='bold')
    ax.tick_params(axis='y', labelsize=14)
    
plt.tight_layout()

q1 = dcopy_new.quantile(0.25)
q3 = dcopy_new.quantile(0.75)
IQR=q3-q1
df_outlier = dcopy_new[(dcopy_new < q3+1.5*IQR) & (dcopy_new > q1-1.5*IQR)]
df_outlier

column_means = df_outlier.mean()
df_outlier_clean = df_outlier.fillna(column_means)
df_outlier_clean

x = df_outlier_clean.duplicated().value_counts()
y = df_outlier_clean.isnull().sum()

print(x)
print()
print(y)

df_outlier_clean_dropduplicate = df_outlier_clean.drop_duplicates(inplace=False)
df_outlier_clean_dropduplicate

dcleancopy = df_outlier_clean_dropduplicate.copy()
dcleancopy

dcleancopy = dcleancopy.astype({"quality":'int'}) 


df_outlier_clean_dropduplicate.info()

!pip install sweetviz
import sweetviz as sv

wine_quality_report = sv.analyze(dcleancopy)
#display the report
wine_quality_report.show_html('wine_report.html')

report = sv.analyze(dcleancopy)

type(df_outlier_clean_dropduplicate["quality"])


df_outlier_clean_dropduplicate["quality"] = df_outlier_clean_dropduplicate["quality"].astype('category', copy=False)
df_outlier_clean_dropduplicate


bins = [0,5,10]


labels = [0, 1] # 'low'=0, 'high'=1

df_outlier_clean_dropduplicate['quality_range']= pd.cut(x=df_outlier_clean_dropduplicate['quality'], bins=bins, labels=labels)

print(df_outlier_clean_dropduplicate[['quality_range','quality']].head(5))
df_outlier_clean_dropduplicate = df_outlier_clean_dropduplicate.drop('quality', axis=1)

fig = px.histogram(df_outlier_clean_dropduplicate, x="quality_range")

fig.show()

correlation = dcopy_new.corr()
plt.figure(figsize=(18, 18))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='Greens')

correlation = df_outlier_clean_dropduplicate.corr()
plt.figure(figsize=(18, 18))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='Greens')

lists = ["fixed acidity" , "volatile acidity"]
for item in lists:

outliers = find_outliers_IQR(dcopy_new[item])
  
X = df_outlier_clean_dropduplicate.drop('quality_range', axis=1)
y = df_outlier_clean_dropduplicate['quality_range']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

X_train.shape, X_test.shape

pca = PCA().fit(X_train_scaled)

loadings = pd.DataFrame(
    data=pca.components_.T * np.sqrt(pca.explained_variance_), 
    columns=[f'PC{i}' for i in range(1, len(X_train.columns) + 1)],
    index=X_train.columns
)
loadings

pc1_loadings = loadings.sort_values(by='PC1', ascending=False)[['PC1']]
pc1_loadings = pc1_loadings.reset_index()
pc1_loadings.columns = ['Attribute', 'CorrelationWithPC1']

plt.bar(x=pc1_loadings['Attribute'], height=pc1_loadings['CorrelationWithPC1'], color='#087E8B')
plt.title('PCA loading scores (first principal component)', size=20)
plt.xticks(rotation='vertical')
plt.show()

algorithms = ['Random Forest', 'Decision Tree', 'Support Vector Machine']
metrics    = ['Confusion Matrix', 'Classification Report','Accuracy']
train_scores = {}
pd.set_option('display.max_rows', 12)

def algorithm_validation(Algorithm=algorithms, Metrics=metrics):        
    if Algorithm == 'Random Forest':
        model = RandomForestClassifier(max_depth=2, random_state=0)
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        X_test['Predict'] = model.predict(X_test)
        
    elif Algorithm == 'Decision Tree':
        model = DecisionTreeClassifier(random_state=0)
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        X_test['Predict'] = model.predict(X_test)
    
    elif Algorithm == 'Support Vector Machine':
        model = SVC(kernel='linear')
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        X_test['Predict'] = model.predict(X_test)
        
    if Metrics == 'Classification Report':
        score = classification_report(y_test, y_pred)
        
    elif Metrics == 'Accuracy':
        score = accuracy_score(y_test, y_pred)
        
    elif Metrics == 'Confusion Matrix':
        plot_confusion_matrix(model, X_test, y_test)
        score = confusion_matrix(y_test, y_pred)
        
    return print('\nThe ' + Metrics + ' of ' + Algorithm + ' is:\n\n'+ str(score) + '\n')
    
algorithms = ['Random Forest', 'Decision Tree', 'Support Vector Machine']
metrics    = ['Confusion Matrix', 'Classification Report','Accuracy']
algorithm_validation('Decision Tree','Classification Report')

algorithms = ['Random Forest', 'Decision Tree', 'Support Vector Machine']
metrics    = ['Confusion Matrix', 'Classification Report','Accuracy']
algorithm_validation('Random Forest','Classification Report')

algorithms = ['Random Forest', 'Decision Tree', 'Support Vector Machine']
metrics    = ['Confusion Matrix', 'Classification Report','Accuracy']
algorithm_validation('Support Vector Machine','Classification Report')

import imblearn
print(imblearn.__version__)

df_outlier_clean_dropduplicate['quality_range'].value_counts()

classifier_tree = DecisionTreeClassifier()

y_predict = classifier_tree.fit(X_train, y_train).predict(X_test)

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

print(classification_report(y_test, y_predict, target_names=["0","1"]))

print(confusion_matrix(y_test, y_predict))

from sklearn import metrics 
from sklearn.metrics import classification_report, confusion_matrix
con_mat = confusion_matrix(y_test, y_predict)
print(con_mat)

sns.heatmap(con_mat, annot=True, fmt="d")
plt.title('Confusion Matrix for Classitification Tree')
plt.xlabel('Predicted')
plt.ylabel('True')

print(classification_report(y_test, y_predict))

 accuracy = metrics.accuracy_score(y_test, y_predict)
 accuracy 
 
 from sklearn import svm

X = df_outlier_clean_dropduplicate.drop('quality_range', axis=1)
y = df_outlier_clean_dropduplicate['quality_range']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10000)

clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn import metrics 
from sklearn.metrics import classification_report, confusion_matrix
con_mat = confusion_matrix(y_test, y_pred)
print(con_mat)

sns.heatmap(con_mat, annot=True, fmt="d")
plt.title('Confusion Matrix for Classitification Tree')
plt.xlabel('Predicted')
plt.ylabel('True')

print(classification_report(y_test, y_pred))

wine_quality_report.show_notebook()