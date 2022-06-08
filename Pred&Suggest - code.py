#!/usr/bin/env python
# coding: utf-8

# # Importing required libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# # Cleaning the collected data

# In[3]:


#data table is the name of the table of data imported through pandas.
data_table = pd.read_csv("feed.csv")


# In[4]:


data_table.head()


# In[5]:


#Replacing the name of 'Date' column to 'Year' 

data_table= data_table.rename(columns={'Date':'Year'})


# In[ ]:





# # Analysing the collected data

# In[6]:


print(data_table.info())


# In[7]:


data_table.describe()


# In[8]:


#The values provided by current sensor:
data_table['AC Cur'].unique()


# In[9]:


# The range of days : 
(data_table['Day'].unique())


# In[10]:


#The duration of data collected (which hour)
sorted(data_table['Hour'].unique())


# In[11]:


#The range of temperature recorded during this duration
temp_range = (data_table['Temp'].unique())


# In[12]:


plt.plot(temp_range,'*')


# ### Hour of the day vs Temperature plot:

# In[13]:


plt.plot(data_table['Hour'],data_table['Temp'],'*',c='red')
plt.xlabel('Hour of the day')
plt.ylabel('Temperature')


# ### Temperature vs Energy Used ( values by current sensor) plot:

# In[14]:


plt.plot(data_table['Temp'],data_table['AC Cur'],'*',c='green')
plt.xlabel('Temperature')
plt.ylabel('Energy Used')


# # Splitting the data (into training and testing sets):

# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


data_table.columns


# In[17]:


X = data_table[['Year', 'Month', 'Day', 'Hour', 'Min', 'Sec','Hum', 'AC Cur']]
y = data_table[['Temp']]


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)


# In[19]:


#sizes of the divided data:

sizes = {'X_train': len(X_train),'y_train':len(y_train),
        'X_test': len(X_test),'y_test':len(y_test)}


# In[20]:


sizes


# In[21]:


##This means that we will train our linear regression model on 67% of total values i.e around 2430 datapoints
## and we will test our linear regression model on 33% of total values i.e. around 1197 datapoints


# In[ ]:





# # Fitting the model on data and testing on test data

# ### Fitting:

# In[22]:


# sklearn is a library and linear model -> sub folder of that library
from sklearn.linear_model import LinearRegression


# In[23]:


model=LinearRegression()


# In[ ]:


model.fit(X_train,y_train) #to train -> "to fit " #we always train a model on training data


# ### Predicting using the fitted model:

# In[25]:


predicted_values = model.predict(X_test) #prediction done on test data without supplying the true values as input


# In[26]:


real_values = np.array(y_test)


# In[27]:


predicted_values


# In[28]:


y


# In[29]:


predicted = predicted_values.tolist()


# In[30]:


predicted


# In[31]:


real_values = real_values.tolist()


# In[32]:


real_values


# In[33]:


plt.scatter(real_values,predicted_values)


# In[ ]:





# In[34]:


df1 = pd.DataFrame(real_values)


# In[35]:


df2 = pd.DataFrame(predicted)


# In[36]:


frames = [df1,df2]


# In[37]:


comparison = pd.concat(frames,axis=1)


# In[38]:


comparison


# In[39]:


comparison.columns = ['Real Value','Predicted Value']


# In[40]:


comparison


# In[41]:


plt.scatter(comparison['Real Value'],comparison['Predicted Value'])


# # Evaluating the model ( Error Percentages)

# In[42]:


from sklearn.metrics import mean_squared_error


# In[43]:


print("Mean Square Error %",mean_squared_error(y_test,predicted_values)*100)


# In[44]:


print("Root Mean Square Error % ",np.sqrt(18.57712464988773))


# In[ ]:





# In[46]:


comparison.head()


# # Suggesting the user:

# In[79]:


trial_real = list(comparison['Real Value'][:40])


# In[80]:


trial_real


# In[81]:


trial_predicted = list(comparison['Predicted Value'][:40])


# In[82]:


trial_predicted


# Assuming that a differnce of 0.5 between real and predicted values is significant in real life, we can suggest the user accordingly

# In[88]:


for i in range(len(trial_real)):
    print(i)
    if(trial_predicted[i]>trial_real[i]):
        if(trial_predicted[i]-trial_real[i]<=0.5):
            print("Nearly similar conditions, please consume energy accordingly")
        elif(trial_predicted[i]-trial_real[i]>0.5):
            print("Can be hotter,can use the AC")
            
    elif(trial_predicted[i]<trial_real[i]):
        if(trial_real[i]-trial_predicted[i]>=0.5):
            print("Can be cooler, using fan is okay")
        elif(trial_real[i]-trial_predicted[i]<0.5):
            print("It's going to be same! Please consume energy accordingly")
            
    
            
 


# In[ ]:




