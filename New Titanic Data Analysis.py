#!/usr/bin/env python
# coding: utf-8

# In[109]:


import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# # Data Dictionary 

# * Survived: 0 = No, 1 = Yes
# * pclass: Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd
# * sibsp: Number of sibilings / spouses aboard the Titanic
# * parch: Number of parents / children aboard the Titanic 
# * ticket: Ticket Number
# * cabin: Cabin Number 
# * embarked: Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton

# In[110]:


train.head()


# In[111]:


test.head()


# In[112]:


train.shape


# In[113]:


test.shape


# In[114]:


train.info()


# There are some missing values in the age column and cabin column 

# In[115]:


test.info()


# There are missing values in the age, fare, and cabin column 

# In[116]:


train.isnull().sum()


# There are 177 rows with a missing Age, 687 rows with a missing Cabin number, and 2 rows with a missing Embarked information. 

# In[117]:


test.isnull().sum()


# There are 87 rows with a missing Age, 327 rows with a missing Cabin number, and 1 row with a missing Fare value.  

# # Import python lib for visualization

# In[118]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()


# In[119]:


def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind = 'bar', stacked = True, figsize = (10,5))


# In[120]:


bar_chart('Sex')


# ###### This shows that women more likely survived than men.

# In[121]:


bar_chart('Pclass')
# 1 = 1st class
# 2 = 2nd class
# 3 = 3rd class


# ###### This shows that 1st class more likely survived than the other classes. However, the 3rd class more likely died than the other classes. 

# In[122]:


bar_chart('SibSp')


# ###### This bar graph shows that people who boarded the ship with no sibilings or spouses were more likely to die than survive. But, people with more than 1 sibiling or spouse more likely survived

# In[123]:


bar_chart('Parch')


# ###### This bar graph shows that people who boarded the ship with more than 2 parents or sibilings more likely survived. 

# In[124]:


bar_chart('Embarked')
# C = Cherbourg
# Q = Queenstown
# S = Southampton


# ###### This bar graph shows that people who boarded the ship from C (Cherbourg), slightly, had a higher chance of surviving. On the other hand, people who boarded the ship from Q (Queenstown) or S (Southampton) more likely died. 

# In[125]:


train.head(10)


# In[126]:


train_test_data = [train, test] # combines the train and test data set

for dataset in train_test_data:
    dataset ['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)


# In[127]:


train['Title'].value_counts()


# In[128]:


test['Title'].value_counts()


# # Directory

# - Mr: 0
# - Miss: 1
# - Mrs: 2
# - Others: 3

# In[129]:


title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2,
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3, "Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona": 3, "Mme": 3, "Capt": 3, "Sir": 3}

for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)


# In[130]:


train.head()


# In[131]:


test.head()


# In[132]:


bar_chart('Title')


# In[133]:


# get rid of unecessary features from the dataset
train.drop('Name', axis = 1, inplace = True)
test.drop('Name', axis = 1, inplace = True)


# In[134]:


train.head()


# In[135]:


test.head()


# # Gender 

# * Male: 0
# * Female: 1

# In[136]:


gender_mapping = {"male": 0, "female": 1}

for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(gender_mapping)


# In[137]:


bar_chart('Sex')


# In[138]:


#fill missing age with median age for each title (Mr, Mrs, Miss, Others)
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace = True)
test["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace = True)


# In[139]:


facet = sns.FacetGrid(train, hue = "Survived", aspect = 4)
facet.map(sns.kdeplot,'Age',shade = True)
facet.set(xlim = (0, train['Age'].max()))
facet.add_legend()

plt.show()


# In[140]:


facet = sns.FacetGrid(train, hue = "Survived", aspect = 4)
facet.map(sns.kdeplot,'Age',shade = True)
facet.set(xlim = (0, train['Age'].max()))
facet.add_legend()

plt.xlim(0, 20)


# People who were 0 to 20 years old had a high chance of surviving. 

# In[141]:


facet = sns.FacetGrid(train, hue = "Survived", aspect = 4)
facet.map(sns.kdeplot,'Age',shade = True)
facet.set(xlim = (0, train['Age'].max()))
facet.add_legend()

plt.xlim(20, 30)


# People who were 20 to 24 had an equal chance of survivng or dieing. However, people over the age of 24 had higher chance of dieing. 

# In[142]:


facet = sns.FacetGrid(train, hue = "Survived", aspect = 4)
facet.map(sns.kdeplot,'Age',shade = True)
facet.set(xlim = (0, train['Age'].max()))
facet.add_legend()

plt.xlim(30, 40)


# People people aged from 30 to 34 had a high chance of dieing but people over the age of 34 had a high chance of surviving. 

# In[143]:


facet = sns.FacetGrid(train, hue = "Survived", aspect = 4)
facet.map(sns.kdeplot,'Age',shade = True)
facet.set(xlim = (0, train['Age'].max()))
facet.add_legend()

plt.xlim(40, 60)


# People of the age group 40 to 60 almost had an equal chance of dieing and surviving. 

# In[144]:


facet = sns.FacetGrid(train, hue = "Survived", aspect = 4)
facet.map(sns.kdeplot,'Age',shade = True)
facet.set(xlim = (0, train['Age'].max()))
facet.add_legend()

plt.xlim(60, 80)


# People of the age group 60 to 80 almost had an equal chance of dieing and surviving. 

# # Binning/Converting Numerical Age to Categorical Variable 

# * Child: 0
# * Young: 1
# * Adult: 2
# * Mid-Age: 3
# * Senior: 4

# In[145]:


for dataset in train_test_data:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0, 
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1, 
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2, 
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3, 
    dataset.loc[(dataset['Age'] > 62), 'Age'] = 4


# In[146]:


train.head()


# In[147]:


bar_chart('Age')
# Child (<=16): 0
# Young (>16 & <=26): 1
# Adult (>26 & <= 36): 2
# Mid-Age (>36 & <=62): 3
# Senior (>62): 4


# Adults had a higher chance of dieing than surviving. Young adults also had a higher hance of dieing. But children seemed to have an equal chance of survivng and dieing. 

# # Embarked

# In[148]:


Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class', '2nd class', '3rd class']
df.plot(kind = 'bar', stacked = True, figsize = (10,5))

# C = Cherbourg
# Q = Queenstown
# S = Southampton


# More than 50% of the 1st class embarked from Southampton. More than 50% of the 2nd class embared from Southampton. More than 50% of the 3rd class embarked from Southampton. In conclusion, 3rd class passengers who embarked from Southampton had a higher chance of surviving amongst all the other classes. 

# In[149]:


# fill out missing embark with S embark 
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# In[150]:


train.head()


# In[151]:


embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)


# # Fare 

# In[152]:


# fill missing Fare with median fare for each Pclass 
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace = True)
test["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace = True)


# In[153]:


facet = sns.FacetGrid(train, hue = "Survived", aspect=4)
facet.map(sns.kdeplot,'Fare', shade = True)
facet.set(xlim = (0, train['Fare'].max()))
facet.add_legend()

plt.show()
# 1 - survived 
# 0 - died 


# ###### People who bought a cheaper ticket had a lower chance of surviving. While the people who bought an expensive ticket had a higher chance of surviving. 

# In[154]:


# lets look at the graph closely
facet = sns.FacetGrid(train, hue = "Survived", aspect=4)
facet.map(sns.kdeplot,'Fare', shade = True)
facet.set(xlim = (0, train['Fare'].max()))
facet.add_legend()
plt.xlim(0, 20)


# ###### People who bought a ticket in the price range of 0-20 had a lower chance of surviving. 

# # Cabin 

# In[48]:


train.Cabin.value_counts()


# In[49]:


for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]


# In[50]:


Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class', '2nd class', '3rd class']
df.plot(kind = 'bar', stacked = True, figsize = (10, 5))


# In[171]:


cabin_mapping = {"A":0, "B":0.4, "C":0.8, "D":1.2, "E":1.6, "F":2, "G":2.4, "T":2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)


# In[172]:


train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace = True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace = True)


# In[173]:


test.head()


# # Family Size

# In[161]:


train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1


# In[162]:


facet = sns.FacetGrid(train, hue = "Survived", aspect = 4)
facet.map(sns.kdeplot, 'FamilySize', shade = True)
facet.set(xlim = (0, train['FamilySize']. max()))
facet.add_legend()
plt.xlim(0)


# In[163]:


family_mapping = {1:0, 2:0.4, 3:0.8, 4:1.2, 5:1.6, 6:2, 7:2.4, 8:2.8, 9:3.2, 10:3.6, 11:4}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)


# In[164]:


train.head()


# In[165]:


# Drop ticket column because it doesn't add value to the data. 
# Drop SibSp and Parch because they are already combined in the FamilySize
features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)


# In[166]:


train_data = train.drop('Survived', axis=1)
target = train['Survived']
train_data.shape, target.shape


# In[167]:


train_data.head(10)


# In[ ]:




