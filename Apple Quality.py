#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv("C:/Users/Admin/Downloads/Excel/apple_quality.csv")


# In[2]:


plt.scatter(df["Sweetness"],df["Crunchiness"])


# In[3]:


plt.plot(df["Sweetness"])
plt.xlabel('Sweetness')
plt.ylabel('Frequency')
plt.title('Line Plot for Apple Quality')
plt.show()


# In[4]:


sns.lineplot(x=df["Juiciness"],y=df["Quality"])
plt.xlabel('Size')
plt.ylabel('Quality')
plt.title('Line Plot for Apple Quality')
plt.show()


# In[5]:


sns.scatterplot(x=df["Crunchiness"], y=df["Ripeness"])
plt.xlabel('Crunchiness')
plt.ylabel('Ripeness')
plt.title('Scatter Plot with Seaborn')
plt.show()


# In[6]:


plt.hist(df["Ripeness"],bins=6,edgecolor="black")
plt.xlabel('Ripeness')
plt.ylabel('Frequency')
plt.title('Histogram with Matplotlib')
plt.show()


# In[7]:


sns.histplot(df["Crunchiness"], bins=5, kde=False, color='lightgreen', edgecolor='black')
plt.xlabel('Crunchiness')
plt.ylabel('Frequency')
plt.title('Histogram with Seaborn')
plt.show()


# In[8]:


sns.boxplot(data=df, color='skyblue')
plt.ylabel('Frequency')
plt.title('Box Plot with Seaborn for Apple Quality')
plt.show()


# In[10]:


import matplotlib.pyplot as plt

sizes = [20, 30, 40, 10]
labels = ['A', 'B', 'C', 'D']
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('Pie Chart')
plt.show()


# In[11]:


df.isnull().sum()


# In[12]:


import seaborn as sns

sns.violinplot(x=df["Juiciness"],y=df["Acidity"])
plt.show()


# In[13]:


import seaborn as sns

sns.pairplot(df)
plt.title('Pair Plot')
plt.show()


# In[14]:


# Drop the 'A_id' and 'Quality' columns from the DataFrame
z=df.drop(['A_id', 'Quality'], axis=1)

# Display the shape of x to confirm the drop
df=z.drop(4000)
df


# In[15]:


df.head()


# In[16]:


df.tail()


# In[17]:


df.describe().transpose()


# In[18]:


# Extract the features (x) and the target variable (y)
x = df.drop("Sweetness",axis=1) # All columns except the last one
y = df["Sweetness"]  # The last column


# In[21]:


from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.2,random_state=42)
x_train.shape,y_train.shape


# In[22]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
model=lr.fit(x_train,y_train)
model


# In[23]:


y_pred=model.predict(x_test)
y_pred


# In[24]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt


# In[25]:


rmse=sqrt(mean_squared_error(y_test,y_pred))
rmse


# In[26]:


r2=r2_score(y_test,y_pred)
r2


# In[27]:


adj_r2=1-float(len(y)-1)/(len(y)-len(lr.coef_)-1)*(1-r2)
adj_r2


# In[ ]:




