
# coding: utf-8

# ### This project is with iris dataset.!

# In[2]:


import pandas as pd
import seaborn as sns


# In[3]:


data = pd.read_csv('C:\\Users\\deeks\\Desktop\\Adsoft\\ML\\Beginners_Level\\Iris1.csv')
data_copy = data.copy()


# In[4]:


data = data.drop(['Unnamed: 5','Unnamed: 6','Unnamed: 7','Unnamed: 8'],axis = 1)


# In[6]:


#print(data.describe())


# In[7]:


data_only = data.drop(['iris'],axis = 1)
#print(data.shape)


# In[15]:


data_label = data['iris']


# ###### Logistic Regression #######

# In[25]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np


# In[16]:


X_train,X_test,y_train,y_test = train_test_split(data_only,data_label,test_size = 0.3,random_state=0)


# In[17]:


print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# In[47]:


classi_logi = LogisticRegression().fit(X_train,y_train)
#classi_logi.predict(X_test)


# In[48]:


y_true = y_test.values
y_predict = classi_logi.predict(X_test)


# ###### Evaluation metrics

# In[37]:


from sklearn.metrics import confusion_matrix,accuracy_score

confusion_matrix(y_true,y_predict,labels = data_label.unique())


# In[38]:


accuracy_score(y_true,y_predict)


# In[43]:


y_true_train = y_train.values
y_predi_train = classi_logi.predict(X_train)


# In[45]:


accuracy_score(y_true_train,y_predi_train)


# ###### Decision Tree
