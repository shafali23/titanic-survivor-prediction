#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


train = pd.read_csv('C:/Users/shafali singal/Desktop/train.csv')
train.head()


# In[6]:


train.info()

train.shape
# In[59]:


train.shape


# In[60]:


train.isnull()


# In[61]:


train.isnull().sum()


# In[62]:


sns.heatmap(train.isnull(), yticklabels = False, cbar= False,  cmap = 'viridis')


# In[63]:



sns.countplot(x= 'Survived',hue= 'Pclass',data=train )


# In[64]:


sns.set_style('whitegrid')


# In[65]:


sns.countplot(x = 'SibSp', data = train)


# In[66]:


train['Age'].hist(bins=40)


# In[67]:


plt.figure(figsize = (10,7))
sns.boxplot(x= 'Pclass',y= 'Age',data=train )


# In[68]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass  == 2:
            return 29
        elif Pclass  == 3:
            return 24
    else:
        return Age
    


# In[69]:


train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis = 1)


# In[70]:


train['Age'].isnull().sum()


# In[71]:


sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# In[72]:


train.drop('Cabin', axis = 1, inplace = True)


# In[73]:


train.head()


# In[74]:


train.Embarked[:-1].unique()


# In[75]:


train.dropna(inplace= True)


# In[76]:


train.head(10)

#after droping values if we again check unique values then it will not show nan
# In[77]:


train.Embarked[:-1].unique()


# In[78]:


train.Sex[:-1].unique()
    


# In[79]:


def value(cols):
    Sex=cols[0]
    if Sex=='male':
        return 0
    else:
        return 1


# In[80]:


train['Sex'] = train[['Sex']].apply(value, axis = 1)


# In[81]:


train['Sex']


# In[82]:


def values(cols):
    Embarked=cols[0]
    if Embarked=='S':
        return 0
    elif Embarked=='C':
        return 1
    else:
        return 2
    
    


# In[83]:


train['Embarked'] = train[['Embarked']].apply(values, axis = 1)


# In[84]:


train['Embarked']


# In[85]:


y=train['Survived']
y


# In[86]:


train.drop(['Survived','Name','Ticket','PassengerId'],axis='columns',inplace=True)


# In[87]:


x=train
x


# In[88]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3)


# In[93]:


from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()


# In[94]:


reg.fit(X_train,y_train)


# In[97]:


z=reg.predict(X_test)
z


# In[98]:


y_test


# In[99]:


y_test-z


# In[ ]:




