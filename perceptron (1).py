#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

df = pd.read_csv (r'train.data')
print (df)
df.head()


# In[3]:


import pandas as pd

df = pd.read_csv (r'test.data')
print (df)
df.head()


# In[4]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


plt.scatter(df['5.0'],df['3.5'],df['1.3'])


# In[7]:


plt.scatter(df['0.3'],df['class-1'])


# In[8]:


X = df[['5.0','3.5','1.3']]


# In[9]:


y= df['0.3']


# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


# In[11]:


X_train


# In[12]:


X_test


# In[13]:


y_train


# In[14]:


y_test


# In[15]:


from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train, y_train)


# In[16]:


X_test


# In[17]:


clf.predict(X_test)


# In[18]:


y_test


# In[19]:


clf.score(X_test, y_test)


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=10)
X_test


# In[ ]:




