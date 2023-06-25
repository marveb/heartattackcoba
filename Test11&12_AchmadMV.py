#!/usr/bin/env python
# coding: utf-8
# # Predicting Heart Attack with Logistic Regression
# In[13]:
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# In[14]:
df = pd.read_csv('C:/Users/LENOVO/PycharmProjects/pythonProject/venv/Scripts/heart.csv')
print(df)

# In[15]:
df.isnull().sum()

# In[16]:
df['output'].value_counts()

# In[28]:
X = df.drop('output', axis=1)
y = df['output']

# setting up testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# ### Predicting Heart Attack
# In[29]:
model = LogisticRegression(solver='liblinear').fit(X_train, y_train)
res = model.predict(X_test)
print(res)

# ### Model Score
# In[30]:
model.score(X_test, y_test)

# ### Heart Attack Probability
# In[31]:
model.predict_proba(X_test)

# make pickle
pickle.dump(model, open("model.pkl", "wb"))
