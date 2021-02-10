#!/usr/bin/env python
# coding: utf-8

# # TASK 1-Prediction using Supervised ML
#  

# # Name- Punarva Gawande

# # GRIPFEB21

# # Task- Predict the percentage of an student based on the no. of study hours.

# In[1]:


#importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#importing data
data =pd.read_csv("C:\\Users\\OWNER\\Desktop\\Student_Score.csv")


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.shape


# In[7]:


plt.boxplot(data)
plt.show()


# # Data Visualization

# In[8]:


plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Hours studies vs Scores')
plt.scatter(data.Hours, data.Scores)
plt.show()


# In[9]:


x = data.iloc[:,:-1].values
y = data.iloc[:,1].values
x


# In[10]:


y


# # Data Preparation

# In[11]:


#import library
from sklearn.model_selection import train_test_split


# In[12]:


x_train, x_test,y_train, y_test = train_test_split(x,y,train_size=0.7, test_size=0.3, random_state=100)


# In[13]:


x_train


# In[14]:


y_train


# In[15]:


x_test


# In[16]:


y_test


# # Linear Regression

# In[17]:


from sklearn.linear_model import LinearRegression


# In[18]:


linreg = LinearRegression()


# In[19]:


linreg.fit(x_train, y_train)


# In[20]:


#plotting regression line
line = linreg.intercept_+ linreg.coef_*x_train


# In[21]:


#plotting for data
plt.scatter (x_train, y_train)
plt.plot(x_train, line)
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Regression Line(Train set)')
plt.show()


# # Test Data

# In[22]:


y_pred=linreg.predict(x_test)
print(y_pred)


# In[23]:


y_test


# In[24]:


#plotting line on test data
plt.scatter (x_test, y_test)
plt.plot(x_test, y_pred)
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Regression Line(Test set)')
plt.show()


# # Actual vs Predicted

# In[25]:


y_test = list(y_test)
prediction = list(y_pred)
data_compare = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
data_compare


# # Accuracy of the Model

# In[26]:


from sklearn import metrics


# In[27]:


metrics.r2_score(y_test, y_pred)


# # Evaluating the Mode

# In[28]:


from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[29]:


print('Mean Squared Error:', metrics.mean_squared_error (y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error (y_test, y_pred))


# # Predicting the Score

# In[30]:


Prediction_score = linreg.predict([[9.25]])
print("predicted score for a student studying 9.25 hours:", Prediction_score)


# # Conclusion
# From the above result we can say that if a student studied for 9.25 hours,then score willbe 93

# In[ ]:




