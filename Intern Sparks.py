#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


print("Pandas version:",pd.__version__)
print("Numpy version:",np.__version__)
print("Matplotlib version:",matplotlib.__version__)
print("sklearn version:",sklearn.__version__)
print("Seaborn version:",sns.__version__)


# In[3]:


data = pd.read_csv('student_scores.csv')
data.head()


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.shape


# In[7]:


data.duplicated().sum()


# In[8]:


data.isnull().sum()


# In[9]:


corr = data.corr()
corr


# In[10]:


sns.heatmap(corr, cmap='Blues',annot=True, fmt='.2g')
plt.show()


# In[11]:


outliers = ['Scores']
plt.rcParams['figure.figsize'] = [6,6]
sns.boxplot(data=data[outliers], orient='v', palette='Set2', width=0.7)
plt.title('Outliers Variable Distribution')
plt.ylabel('Profit Range')
plt.xlabel('Continuous Variable')
plt.show()


# In[12]:


sns.displot(data['Scores'], bins=5, kde=True)
plt.show()


# In[13]:


sns.pairplot(data)


# In[14]:


X = data['Hours'].values.reshape(-1,1)
y = data['Scores'].values.reshape(-1,1)


# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 5)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[16]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,y)


# In[17]:


line = lr.coef_ * X + lr.intercept_


# In[18]:


plt.scatter(X_train , y_train , color = "#329ba8")
plt.plot(X , line , color = "r")
plt.show()


# In[19]:


y_pred = lr.predict(X_test)
y_pred


# In[20]:


plt.scatter(X_test,y_test , color = "#75a6eb")
plt.plot(X_test,y_pred , color = "black")
plt.show()


# In[ ]:




