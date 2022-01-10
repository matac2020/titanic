#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import copy
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Algoritma Paketleri#
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
from sklearn import model_selection
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from math import sqrt
import math
from matplotlib import pyplot
from sklearn.utils import resample
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from ipywidgets import Image
from io import StringIO
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
get_ipython().system('pip install pydotplus')
import pydotplus
get_ipython().system('pip install graphviz')
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from sklearn import tree
from sklearn import ensemble
from sklearn.metrics import precision_recall_curve
from IPython.display import Image
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
get_ipython().system('pip install xgboost')
get_ipython().system('pip install lightgbm')
get_ipython().system('pip install catboost')


# In[3]:


from sklearn.cluster import KMeans


# In[4]:


from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor
from sklearn import decomposition, datasets
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import randint
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.manifold import TSNE
from sklearn.metrics import make_scorer


# In[5]:


train=pd.read_csv("/Users/mac/Desktop/AI_Works/Titanic/train.csv")


# In[6]:


test= pd.read_csv("/Users/mac/Desktop/AI_Works/Titanic/test.csv")


# In[8]:


train.head()


# In[9]:


train.info()


# In[10]:


train["Pclass"].unique()


# In[11]:


train["SibSp"].unique()


# In[12]:


train["Parch"].unique()


# In[13]:


train["Embarked"].unique()


# In[14]:


pc_class=pd.pivot_table(train,index=["Pclass"],values=["Survived"],aggfunc={"Survived":["count","sum"]})

pc_class["y_avg"]= pc_class["Survived"]["sum"]*100 / pc_class["Survived"]["count"]

pc_class.sort_values(by="y_avg",ascending=True)


# In[15]:


pc_class=pd.pivot_table(train,index=["Sex"],values=["Survived"],aggfunc={"Survived":["count","sum"]})

pc_class["y_avg"]= pc_class["Survived"]["sum"]*100 / pc_class["Survived"]["count"]

pc_class.sort_values(by="y_avg",ascending=True)


# In[16]:


pc_class=pd.pivot_table(train,index=["Age"],values=["Survived"],aggfunc={"Survived":["count","sum"]})

pc_class["y_avg"]= pc_class["Survived"]["sum"]*100 / pc_class["Survived"]["count"]

pc_class.sort_values(by="y_avg",ascending=True)


# In[17]:


train["Age"].corr(train["Survived"])


# In[18]:


pc_class=pd.pivot_table(train,index=["Sex"],values=["Survived"],aggfunc={"Survived":["count","sum"]})

pc_class["y_avg"]= pc_class["Survived"]["sum"]*100 / pc_class["Survived"]["count"]

pc_class.sort_values(by="y_avg",ascending=True)


# In[19]:


pc_class=pd.pivot_table(train,index=["SibSp"],values=["Survived"],aggfunc={"Survived":["count","sum"]})

pc_class["y_avg"]= pc_class["Survived"]["sum"]*100 / pc_class["Survived"]["count"]

pc_class.sort_values(by="y_avg",ascending=True)


# In[20]:


train["SibSp"].corr(train["Survived"])


# In[21]:


pc_class=pd.pivot_table(train,index=["Parch"],values=["Survived"],aggfunc={"Survived":["count","sum"]})

pc_class["y_avg"]= pc_class["Survived"]["sum"]*100 / pc_class["Survived"]["count"]

pc_class.sort_values(by="y_avg",ascending=True)


# In[22]:


train["Parch"].corr(train["Survived"])


# In[23]:


pc_class=pd.pivot_table(train,index=["Cabin"],values=["Survived"],aggfunc={"Survived":["count","sum"]})

pc_class["y_avg"]= pc_class["Survived"]["sum"]*100 / pc_class["Survived"]["count"]

pc_class.sort_values(by="y_avg",ascending=True)


# In[24]:


pc_class=pd.pivot_table(train,index=["Embarked"],values=["Survived"],aggfunc={"Survived":["count","sum"]})

pc_class["y_avg"]= pc_class["Survived"]["sum"]*100 / pc_class["Survived"]["count"]

pc_class.sort_values(by="y_avg",ascending=True)


# In[25]:


pc_class=pd.pivot_table(train,index=["Fare"],values=["Survived"],aggfunc={"Survived":["count","sum"]})

pc_class["y_avg"]= pc_class["Survived"]["sum"]*100 / pc_class["Survived"]["count"]

pc_class.sort_values(by="y_avg",ascending=True)


# In[26]:


train["Fare"].corr(train["Survived"])


# In[27]:


from quickda.explore_data import *
from quickda.clean_data import *
from quickda.explore_numeric import *
from quickda.explore_categoric import *
from quickda.explore_numeric_categoric import *
from quickda.explore_time_series import *


# In[28]:


explore(train)


# In[29]:


train.info()


# In[36]:


explore(train, method="profile", report_name="Titanic Investigation")


# In[37]:


A = [3, 6, 2, 7, 2, 7, 1, 6, 3]
for i in range(0, len(A)):
    key = A[i]
    j = i - 1
    while j >=0 and A[j] > key:
        A[j + 1] = A[j]
        j = j - 1
    A[j + 1] = key
    
print(A)


# In[30]:


train.info()


# In[31]:


train[train["Age"].isnull()]["Survived"].mean()


# In[32]:


train[train["Cabin"].isnull()]["Survived"].mean()


# In[33]:


train= clean(train,method="standardize")


# In[34]:


train.info()


# In[35]:


train["survived"].mean()


# In[36]:


train["pclass"].unique()


# In[37]:


train["age"].fillna(150,inplace=True)


# In[38]:


train["age_group"]=["<20" if x<20 else "<82" if x<82 else "null" for x in train["age"]]

Segment_1=pd.pivot_table(train,index=["age_group"],values=["survived"],aggfunc={"survived":["count","sum"]})

Segment_1["y_avg"]= Segment_1["survived"]["sum"]*100 / Segment_1["survived"]["count"]

Segment_1.sort_values(by="y_avg",ascending=True)


# In[39]:


train["sibsp"].unique()


# In[40]:


train["sibs_group"]=["zero" if x<1 else "<3" if x<3 else "3+" for x in train["sibsp"]]

Segment_1=pd.pivot_table(train,index=["sibs_group"],values=["survived"],aggfunc={"survived":["count","sum"]})

Segment_1["y_avg"]= Segment_1["survived"]["sum"]*100 / Segment_1["survived"]["count"]

Segment_1.sort_values(by="y_avg",ascending=True)


# In[41]:


train["parch"].unique()


# In[42]:


train["parch_group"]=["zero" if x<1 else "<4" if x<4 else "4+" for x in train["parch"]]

Segment_1=pd.pivot_table(train,index=["parch_group"],values=["survived"],aggfunc={"survived":["count","sum"]})

Segment_1["y_avg"]= Segment_1["survived"]["sum"]*100 / Segment_1["survived"]["count"]

Segment_1.sort_values(by="y_avg",ascending=True)


# In[43]:


train["fare"].describe()


# In[44]:


train["fare_group"]=["<12" if x<12 else "<51" if x<51 else "51+" for x in train["fare"]]

Segment_1=pd.pivot_table(train,index=["fare_group"],values=["survived"],aggfunc={"survived":["count","sum"]})

Segment_1["y_avg"]= Segment_1["survived"]["sum"]*100 / Segment_1["survived"]["count"]

Segment_1.sort_values(by="y_avg",ascending=True)


# In[45]:


train["embarked"].unique()


# In[46]:


train[train["embarked"].isnull()]["survived"].mean()


# In[47]:


train["embarked"].fillna("null",inplace=True)


# In[48]:


Segment_1=pd.pivot_table(train,index=["embarked"],values=["survived"],aggfunc={"survived":["count","sum"]})

Segment_1["y_avg"]= Segment_1["survived"]["sum"]*100 / Segment_1["survived"]["count"]

Segment_1.sort_values(by="y_avg",ascending=True)


# In[49]:


Segment_1=pd.pivot_table(train,index=["sex"],values=["survived"],aggfunc={"survived":["count","sum"]})

Segment_1["y_avg"]= Segment_1["survived"]["sum"]*100 / Segment_1["survived"]["count"]

Segment_1.sort_values(by="y_avg",ascending=True)


# In[50]:


train.info()


# In[51]:


train.drop(columns=["passengerid","pclass","name","age","sibsp","parch","ticket","fare","cabin"],inplace=True)


# In[52]:


train.head()


# In[53]:


y=train[["survived"]]


# In[54]:


x=train[["survived","sex","embarked","age_group","sibs_group","parch_group","fare_group"]]


# In[55]:


x.head()


# In[56]:


x.info()


# In[57]:


cat_vars = x.select_dtypes(include=['object']).copy().columns


# In[58]:


for var in  cat_vars:
        # for each cat add dummy var, drop original column
        x = pd.concat([x.drop(var, axis=1), pd.get_dummies(x[var], prefix=var, prefix_sep='_', drop_first=True)], axis=1)
    


# In[59]:


x.head()


# In[60]:


x.drop(columns=("survived"),inplace=True)


# In[61]:


x.info()


# In[62]:


y.head()


# In[63]:


pip install pyforest


# In[72]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=2021)


# In[75]:


#Decision Tree#
df_train, df_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2020,stratify=y)
clf = DecisionTreeClassifier()
clf = clf.fit(df_train,y_train)
y_pred = clf.predict(df_test)
print(cm(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[76]:


#AdaBoost#
df_train, df_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2020,stratify=y)
AdaBoost = AdaBoostClassifier(n_estimators=50,learning_rate=1)
AdaBoost.fit(df_train,y_train)
prediction = AdaBoost.score(df_test,y_test)
print('The accuracy is: ',prediction*100,'%')
abc=AdaBoost.predict(df_test)
print(cm(y_test, abc))
print(classification_report(y_test, abc))


# In[77]:


#Gradient Boosting#
sgb = GradientBoostingClassifier(n_estimators=50, random_state=2020,learning_rate=0.1,subsample=0.5)

sgb = sgb.fit(df_train,y_train)

#Predict the response for test dataset
y_pred = sgb.predict(df_test)
y_pred_prob = sgb.predict_proba(df_test)[:,1]

print(cm(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

feature_importances = pd.DataFrame(sgb.feature_importances_,
                                   index = df_train.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)

print(feature_importances)


# In[78]:


#Random Forest#
df_train, df_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=2020,stratify=y)
rf = RandomForestClassifier(n_estimators=4, max_depth = 5, random_state=2020)
rf.fit(df_train, y_train)
predictions = rf.predict(df_test)
score = round(accuracy_score(y_test, predictions), 3)
cm1 = cm(y_test, predictions)
sns.heatmap(cm1, annot=True, fmt=".0f")
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Accuracy Score: {0}'.format(score), size = 15)
plt.show()

print(cm(y_test, predictions))
print("Accuracy:",metrics.accuracy_score(y_test, predictions))

print(classification_report(y_test, predictions))


# In[79]:


df= copy.deepcopy(x)
df.columns = df.columns.str.replace("<", "kuçuk")


# In[82]:


#SVM#
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2020,stratify=y)
svc = SVC()
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

svc.score(X_test, y_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# ## PART 2-How to automatize sub-grouping?

# In[154]:


t2=pd.read_csv("/Users/mac/Desktop/AI_Works/Titanic/train.csv")


# In[155]:


t2.head()


# In[156]:


t2.info()


# In[157]:


t2.drop("Cabin",axis=1,inplace=True)


# In[158]:


t2.info()


# In[159]:


t2.dropna(how="any",inplace=True)


# In[160]:


t2.info()


# In[161]:


t2.head()


# In[101]:


model= KMeans(n_clusters=3)


# In[103]:


t2.select_dtypes(include=["float","int"]).columns


# In[105]:


u= t2[['PassengerId','Age', 'SibSp', 'Parch', 'Fare']]


# In[109]:


t2.shape


# In[118]:


l.columns


# In[108]:


f=t2[["Age","Survived"]]
c=model.fit_predict(t2[["Age","Survived"]])
d=pd.DataFrame(c)
d=d.reset_index()
e=pd.concat([d,f],axis=1,sort=False)
e


# In[ ]:


l=pd.DataFrame(c)
l=l.reset_index()
l.drop("index",axis=1,inplace=True)
l.rename(columns={0: "sfdgg"})


# In[172]:


type(t2)


# In[212]:


t2.shape


# In[204]:


p.shape


# In[214]:


t2


# In[215]:


t5=copy.deepcopy(t2.reset_index())


# In[216]:


t5


# In[101]:


model= KMeans(n_clusters=3)


# In[217]:


for o in u.columns:
    p= model.fit_predict(t2[[o]])
    p=pd.DataFrame(p)
    p=p.reset_index()
    p.drop("index",axis=1,inplace=True)
    p.rename(columns={0: o+"_New"},inplace=True)
    t5=pd.concat([t5,p],axis=1,sort=False)


# In[218]:


print(t5.shape)
t5.head()


# ### Grid Search & Random Search w Decision Tree

# In[72]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=2021)


# In[423]:


#Decision Tree#
df_train, df_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2020,stratify=y)
clf = DecisionTreeClassifier()
clf = clf.fit(df_train,y_train)
y_pred = clf.predict(df_test)
print(cm(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# #Decision Tree#
# df_train, df_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2020,stratify=y)
# clf = DecisionTreeClassifier()
# clf = clf.fit(df_train,y_train)
# y_pred = clf.predict(df_test)
# print(cm(y_test, y_pred))
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print('Mean of Accuracy:',(cross_val_score(clf, x, y, cv=4, scoring='accuracy')).mean())

# In[220]:


help(DecisionTreeClassifier())


# In[221]:


clf.get_params()


# In[ ]:


#Random Search#


# In[230]:


from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint


# In[224]:


criterion = ["gini", "entropy"]


# In[225]:


splitter = ["best", "random"]


# In[388]:


max_depth=["None",1,2,3,4,5,6,7,14,16]


# In[271]:


min_samples_split =[int(x) for x in np.linspace(start=2,stop=10,num=8)]


# In[272]:


splitter = ["best","random"]


# In[316]:


min_samples_leaf= [int(x) for x in np.linspace(1,5,num=4)]


# In[389]:


random_grid = {'criterion':criterion,'splitter':splitter,'max_depth':max_depth,'min_samples_split':min_samples_split,'splitter':splitter,'min_samples_leaf':min_samples_leaf}


# In[390]:


pprint(random_grid)


# In[391]:


clf = DecisionTreeClassifier()


# In[392]:


df_train, df_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2020,stratify=y)


# In[393]:


clf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid,n_iter=250,
                               cv = 4, random_state=2020)


# In[394]:


clf_random.fit(df_train,y_train)


# In[395]:


clf_random.best_params_


# In[396]:


clf_random.best_estimator_


# In[397]:


y_predict=clf_random.best_estimator_.predict(df_test)


# In[398]:


print(cm(y_test, y_predict))


# In[399]:


print("Accuracy:",metrics.accuracy_score(y_test, y_predict))


# In[400]:


print(classification_report(y_test, y_predict))


# In[401]:


#Grid Search#


# In[402]:


from sklearn.model_selection import GridSearchCV


# In[403]:


param_grid = {'criterion':criterion,'splitter':splitter,'max_depth':max_depth,'min_samples_split':min_samples_split,'splitter':splitter,'min_samples_leaf':min_samples_leaf}


# In[404]:


clf = DecisionTreeClassifier()


# In[405]:


df_train, df_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2020,stratify=y)


# In[406]:


clf_grid = GridSearchCV(estimator = clf, param_grid = param_grid,
                               cv = 4)


# In[407]:


clf_grid.fit(df_train,y_train)


# In[408]:


clf_grid.best_params_


# In[409]:


clf_grid.best_estimator_


# In[410]:


y_predict=clf_grid.best_estimator_.predict(df_test)


# In[411]:


print(cm(y_test, y_predict))


# In[414]:


print("Accuracy:",metrics.accuracy_score(y_test, y_predict))


# In[413]:


print(classification_report(y_test, y_predict))


# In[ ]:


#Gender extraction by using RegEx#


# In[65]:


t3=pd.read_csv("/Users/mac/Desktop/AI_Works/Titanic/train.csv")


# In[66]:


t3.head()


# In[67]:


t3.info()


# In[68]:


t3.drop("Cabin",axis=1,inplace=True)


# In[70]:


t3.info()


# In[73]:


t3.dropna(how="any",inplace=True)


# In[74]:


t3.info()


# In[75]:


t3.head()


# In[109]:


t3.Sex.value_counts()


# In[101]:


t3.Name.head(15)


# In[139]:


t3.reset_index(inplace=True)


# In[142]:


t3.head(10)


# In[148]:


t3["Sex"][5]


# In[163]:


t4=t3.Name


# In[166]:


t4[6]


# In[80]:


import re


# In[215]:


regex1="Miss."


# In[216]:


regex2="Mrs."


# In[217]:


regex3= "Mr."


# In[218]:


regex4="Master."


# In[219]:


regex5="Ms."


# In[220]:


regex6="Dr."


# In[221]:


regex7="Rev."


# In[222]:


regex8="Major."


# In[223]:


regex9="Col."


# In[225]:


regex10="Mlle."


# In[226]:


regex11="Countess."


# In[230]:


regex12="Don."


# In[231]:


regex13="Mme."


# In[232]:


regex14="Capt."


# In[237]:


regex15="Jonkheer."


# In[238]:


t3["new_sex"]=0


# In[239]:


for i in range(len(t4)):
    if re.search(regex1,t4[i]) != None:
        t3["new_sex"][i] ="female"
    elif re.search(regex2,t4[i]) != None:
        t3["new_sex"][i]="female"
    elif re.search(regex3,t4[i]) !=None:
        t3["new_sex"][i]="male"
    elif re.search(regex4,t4[i]) !=None:
        t3["new_sex"][i]="male"
    elif re.search(regex5,t4[i]) != None:
        t3["new_sex"][i] ="female"
    elif re.search(regex6,t4[i]) != None:
        t3["new_sex"][i]="male"
    elif re.search(regex7,t4[i]) !=None:
        t3["new_sex"][i]="male"
    elif re.search(regex8,t4[i]) !=None:
        t3["new_sex"][i]="male"
    elif re.search(regex9,t4[i]) != None:
        t3["new_sex"][i]="male"
    elif re.search(regex10,t4[i]) !=None:
        t3["new_sex"][i]="female"
    elif re.search(regex11,t4[i]) !=None:
        t3["new_sex"][i]="female"
    elif re.search(regex12,t4[i]) != None:
        t3["new_sex"][i] ="male"
    elif re.search(regex13,t4[i]) != None:
        t3["new_sex"][i]="female"
    elif re.search(regex14,t4[i]) !=None:
        t3["new_sex"][i]="male"
    elif re.search(regex15,t4[i]) !=None:
        t3["new_sex"][i]="male"
    else:
        t3["new_sex"][i]="Error"


# In[240]:


t3


# In[241]:


t3[t3["new_sex"]=="Error"]


# In[245]:


def g(row):
    if row["Sex"] == row["new_sex"]:
        val = 0
    else:
        val= 1
    return val

t3["check"]= t3.apply(g,axis=1)


# In[249]:


t3["check"].sum()


# In[250]:


t3[t3["check"]==1]


# In[257]:


def factorial(n):
  
  # the base case that tells us to stop
  if n ==0:
    return 1
  
  # the recursive case that calls itself
  else:
    return n * factorial(n-1)
   # recursion is calling the function on simpler input


# In[258]:


factorial(5)


# In[ ]:


#FEATURE SCALING > Standardization&Normalization


# In[13]:


t3=pd.read_csv("/Users/mac/Desktop/AI_Works/Titanic/train.csv")


# In[14]:


t3.head()


# In[15]:


t3.info()


# In[16]:


t3.drop("Cabin",axis=1,inplace=True)


# In[17]:


t3.info()


# In[18]:


t3.dropna(how="any",inplace=True)


# In[19]:


t3.info()


# In[20]:


t3.head()


# In[22]:


t4 = t3.drop(columns=['PassengerId','Pclass', 'Name','Sex','Ticket','Embarked'])


# In[23]:


t4.columns


# In[64]:


y=t4[["Survived"]]


# In[65]:


x=t4[['Age', 'SibSp', 'Parch', 'Fare']]


# In[27]:


#Decision Tree#
df_train, df_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2020,stratify=y)
clf = DecisionTreeClassifier()
clf = clf.fit(df_train,y_train)
y_pred = clf.predict(df_test)
print(cm(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[28]:


#AdaBoost#
df_train, df_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2020,stratify=y)
AdaBoost = AdaBoostClassifier(n_estimators=50,learning_rate=1)
AdaBoost.fit(df_train,y_train)
prediction = AdaBoost.score(df_test,y_test)
print('The accuracy is: ',prediction*100,'%')
abc=AdaBoost.predict(df_test)
print(cm(y_test, abc))
print(classification_report(y_test, abc))


# In[29]:


#Gradient Boosting#
sgb = GradientBoostingClassifier(n_estimators=50, random_state=2020,learning_rate=0.1,subsample=0.5)

sgb = sgb.fit(df_train,y_train)

#Predict the response for test dataset
y_pred = sgb.predict(df_test)
y_pred_prob = sgb.predict_proba(df_test)[:,1]

print(cm(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

feature_importances = pd.DataFrame(sgb.feature_importances_,
                                   index = df_train.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)

print(feature_importances)


# In[30]:


#Random Forest#
df_train, df_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=2020,stratify=y)
rf = RandomForestClassifier(n_estimators=4, max_depth = 5, random_state=2020)
rf.fit(df_train, y_train)
predictions = rf.predict(df_test)
score = round(accuracy_score(y_test, predictions), 3)
cm1 = cm(y_test, predictions)
sns.heatmap(cm1, annot=True, fmt=".0f")
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Accuracy Score: {0}'.format(score), size = 15)
plt.show()

print(cm(y_test, predictions))
print("Accuracy:",metrics.accuracy_score(y_test, predictions))

print(classification_report(y_test, predictions))


# In[31]:


df= copy.deepcopy(x)
df.columns = df.columns.str.replace("<", "kuçuk")


# In[32]:


#SVM#
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2020,stratify=y)
svc = SVC()
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

svc.score(X_test, y_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:


#APPLYING STANDARDIZATION


# In[33]:


from sklearn.preprocessing import StandardScaler


# In[38]:


std_scale = StandardScaler().fit(x[['Age', 'SibSp', 'Parch', 'Fare']])


# In[39]:


df_std = std_scale.transform(x[['Age', 'SibSp', 'Parch', 'Fare']])


# In[43]:


type(df_std)


# In[53]:


x= pd.DataFrame(df_std)


# In[54]:


type(x)


# In[55]:


#Decision Tree#
df_train, df_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2020,stratify=y)
clf = DecisionTreeClassifier()
clf = clf.fit(df_train,y_train)
y_pred = clf.predict(df_test)
print(cm(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[56]:


#AdaBoost#
df_train, df_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2020,stratify=y)
AdaBoost = AdaBoostClassifier(n_estimators=50,learning_rate=1)
AdaBoost.fit(df_train,y_train)
prediction = AdaBoost.score(df_test,y_test)
print('The accuracy is: ',prediction*100,'%')
abc=AdaBoost.predict(df_test)
print(cm(y_test, abc))
print(classification_report(y_test, abc))


# In[57]:


#Gradient Boosting#
sgb = GradientBoostingClassifier(n_estimators=50, random_state=2020,learning_rate=0.1,subsample=0.5)

sgb = sgb.fit(df_train,y_train)

#Predict the response for test dataset
y_pred = sgb.predict(df_test)
y_pred_prob = sgb.predict_proba(df_test)[:,1]

print(cm(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

feature_importances = pd.DataFrame(sgb.feature_importances_,
                                   index = df_train.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)

print(feature_importances)


# In[58]:


#Random Forest#
df_train, df_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=2020,stratify=y)
rf = RandomForestClassifier(n_estimators=4, max_depth = 5, random_state=2020)
rf.fit(df_train, y_train)
predictions = rf.predict(df_test)
score = round(accuracy_score(y_test, predictions), 3)
cm1 = cm(y_test, predictions)
sns.heatmap(cm1, annot=True, fmt=".0f")
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Accuracy Score: {0}'.format(score), size = 15)
plt.show()

print(cm(y_test, predictions))
print("Accuracy:",metrics.accuracy_score(y_test, predictions))

print(classification_report(y_test, predictions))


# In[ ]:


df= copy.deepcopy(x)
df.columns = df.columns.str.replace("<", "kuçuk")


# In[60]:


#SVM#
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2020,stratify=y)
svc = SVC()
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

svc.score(X_test, y_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:


#APPLYING NORMALIZATION


# In[61]:


from sklearn.preprocessing import MinMaxScaler


# In[66]:


normal_scale = MinMaxScaler().fit(x[['Age', 'SibSp', 'Parch', 'Fare']])


# In[67]:


df_nor = normal_scale.transform(x[['Age', 'SibSp', 'Parch', 'Fare']])


# In[68]:


type(df_nor)


# In[69]:


x= pd.DataFrame(df_nor)


# In[70]:


type(x)


# In[71]:


#Decision Tree#
df_train, df_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2020,stratify=y)
clf = DecisionTreeClassifier()
clf = clf.fit(df_train,y_train)
y_pred = clf.predict(df_test)
print(cm(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[72]:


#AdaBoost#
df_train, df_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2020,stratify=y)
AdaBoost = AdaBoostClassifier(n_estimators=50,learning_rate=1)
AdaBoost.fit(df_train,y_train)
prediction = AdaBoost.score(df_test,y_test)
print('The accuracy is: ',prediction*100,'%')
abc=AdaBoost.predict(df_test)
print(cm(y_test, abc))
print(classification_report(y_test, abc))


# In[73]:


#Gradient Boosting#
sgb = GradientBoostingClassifier(n_estimators=50, random_state=2020,learning_rate=0.1,subsample=0.5)

sgb = sgb.fit(df_train,y_train)

#Predict the response for test dataset
y_pred = sgb.predict(df_test)
y_pred_prob = sgb.predict_proba(df_test)[:,1]

print(cm(y_test, y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

feature_importances = pd.DataFrame(sgb.feature_importances_,
                                   index = df_train.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)

print(feature_importances)


# In[74]:


#Random Forest#
df_train, df_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=2020,stratify=y)
rf = RandomForestClassifier(n_estimators=4, max_depth = 5, random_state=2020)
rf.fit(df_train, y_train)
predictions = rf.predict(df_test)
score = round(accuracy_score(y_test, predictions), 3)
cm1 = cm(y_test, predictions)
sns.heatmap(cm1, annot=True, fmt=".0f")
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Accuracy Score: {0}'.format(score), size = 15)
plt.show()

print(cm(y_test, predictions))
print("Accuracy:",metrics.accuracy_score(y_test, predictions))

print(classification_report(y_test, predictions))


# In[ ]:


df= copy.deepcopy(x)
df.columns = df.columns.str.replace("<", "kuçuk")


# In[76]:


#SVM#
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2020,stratify=y)
svc = SVC()
svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

svc.score(X_test, y_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:




