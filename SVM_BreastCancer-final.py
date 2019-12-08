
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astral import Astral
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import linear_model

from sklearn.model_selection import cross_val_score
from sklearn import datasets


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
import seaborn as sns


# In[6]:


dataset = pd.read_csv('/Users/agatakaczmarek/Data_Science/SVM/data.csv', sep=',')


# # Exploring dataset, null values and correlation  between variables

# In[7]:


dataset.head()


# In[8]:


dataset.info()


# Checking if the dataset has any null values

# In[9]:


dataset.isnull()


# Null values found in the 32th column, after investigating it is clear the 32 column is completely blank, probably a badly formatted csv file. So I am dropping the column from the dataset

# In[15]:


dataset= dataset.drop(dataset.columns[32], axis=1)


# In[16]:


dataset.info()


# In[17]:


dataset.isnull()


# I would like to see if there is any relationship between pairs of variables and the diagnosis. I am choosing first 5 variables.

# In[20]:

# I want to check if there any any relationships between factors. It will be useful to visual data to see if it can be linearly seperable.
# Although not the final and most accurate measure of linear separability, it can suggest to us if we can use Linear kernel for classification.
# At those examples I can see that perhaps not all the factors would be linearly separable.

#Potential relationship exists therefore I want to investigate it further with the correlation Matrix

sns.pairplot(dataset, hue = "diagnosis", vars = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean"])


# In[23]:

plt.figure(figsize=(20,15))
sns.heatmap (dataset.corr(), annot= True)


# There are few strong correlations apart from the more obvious like radius mean and radius worst, more interesting is strong correlation between mean radius and mean perimeter, as well as mean area and mean perimeter

# In[47]:


X= dataset.drop (["diagnosis", "id"], axis=1)


# In[48]:


X.head()


# In[49]:


y = dataset ["diagnosis"]


# In[50]:


y.head()


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25)


# In[52]:


X_train.shape


# In[53]:


X_test.shape


# In[54]:


y_train.shape


# In[55]:


y_test.shape


# These are my training and testing data. For the training we have 398 observatios and for testing 171.

# # Smoothing features with normalization

# Variables are all numerical but in a very different scales, so it is important to bring them to a more normalized form before training the model.

# In[56]:


X_train_min= X_train.min()
X_train.min


# In[57]:


X_train_max= X_train.max()
X_train.max


# In[58]:


X_train_range = (X_train_max -X_train_min)
X_train_range


# In[59]:


X_train_scaled= (X_train -X_train_min)/X_train_range
X_train_scaled.head()


# In[60]:


X_test_min = X_test.min()
X_test_range= (X_test - X_test_min).max()
X_test_scaled= (X_test - X_test_min)/X_test_range


# In[69]:
# # Fitting a SVC model
# Since there are different SVC models to choose from, it is important to recognise the differences between them. 
#I will first try the general linear kerner with a C value of 1. Data is normalised so often a general kernel with a standard parameter C = 1 could sufficle. 


C = 1.0
svc = svm.SVC(kernel='linear', C=C)


# In[70]:


svc.fit(X_train_scaled, y_train)


# In[71]:


y_predict= svc.predict(X_test_scaled)


# # Next step is to check the accuracy of our prediction. We are going to use confusion matrix for this comparison.

# In[72]:


from sklearn.metrics import classification_report, confusion_matrix


# In[73]:


cm= confusion_matrix(y_test,y_predict)


# In[75]:


cm=np.array(confusion_matrix(y_test, y_predict))
confusion= pd.DataFrame(cm, index=["is_cancer", "is_healthy"],columns= ["predicted_cancer", "predicted_healthy"])
confusion


# In[76]:


sns.heatmap(confusion, annot=True)


# In[77]:


print(classification_report(y_test,y_predict))


# #  Models Accuracy analysis
# SVC used C=1, Linear Kernel gives reasonable results and based on the confusion matrix, it only predicts one time that the tumor is cancerous while it was actually healthy. 
#More concerning is the fact that it predicted 9 times that the tumor is healthy while in fact it was cancerous. All in all it give around 94% accuracy.

# # We can try to improve model accuracy with tweaking model parameters.
# I used SVC Linear kernel before. Now I will try to use more sophisticated  model which can map it to a higher dimensional space like a Gaussian kernel. 
#The standard linear SVM has low bias and high variance, but the trade-off can be changed by increasing the C parameter that influences the number of violations of the margin allowed in the training data 
#which increases the bias but decreases the variance.

#%% Trying the same linear kernel but with unscaled data for comparison

svc.fit(X_train, y_train)


# In[71]:


y_predict= svc.predict(X_test)


# # Next step is to check the accuracy of our prediction. We are going to use confusion matrix for this comparison.



# In[73]:


cm= confusion_matrix(y_test,y_predict)


# In[75]:


cm=np.array(confusion_matrix(y_test, y_predict))
confusion= pd.DataFrame(cm, index=["is_cancer", "is_healthy"],columns= ["predicted_cancer", "predicted_healthy"])
confusion


# In[76]:


sns.heatmap(confusion, annot=True)


# In[77]:


print(classification_report(y_test,y_predict))

# Unsuprisingly the unscaled dataset performed worst in terms of accuracy.
#So i will continue to work with the scaled set of data.


#%% I will try a more complex Gaussian Kernel. The regular C value of 1 and a regular value of gamma of 1 gives good results of 0.93 
#but it is not better than the linear model


from sklearn.svm import SVC

C = 1
gamma=1
svclassifier = svm.SVC(kernel='rbf',C=C, gamma= gamma)  


svclassifier.fit(X_train_scaled, y_train)  

y_predict = svclassifier.predict(X_test_scaled)  


# In[86]:


cm= confusion_matrix(y_test,y_predict)
cm=np.array(confusion_matrix(y_test, y_predict))
confusion= pd.DataFrame(cm, index=["is_cancer", "is_healthy"],columns= ["predicted_cancer", "predicted_healthy"])

sns.heatmap(confusion, annot=True)
print(classification_report(y_test,y_predict))


#%% The kernel functions can be seen as an efficient way to transform your original features into another space,
#where a separating hyperplane in the new feature space does not have to be linear in the original feature space.
# The gamma parameter is used for the Gaussian kernel function, The higher the gamma value it tries to exactly fit the training data set.
#So I will try to adjust the gamma parameter of the Gaussian function and see it it works better.
## If I increase the value of C twice I can make the accuracy better, but inreasing gamma didnt have any positive effects.


from sklearn.svm import SVC

C = 2
gamma=1
svclassifier = svm.SVC(kernel='rbf',C=C, gamma= gamma)  


svclassifier.fit(X_train_scaled, y_train)  

y_predict = svclassifier.predict(X_test_scaled)  

cm= confusion_matrix(y_test,y_predict)
cm=np.array(confusion_matrix(y_test, y_predict))
confusion= pd.DataFrame(cm, index=["is_cancer", "is_healthy"],columns= ["predicted_cancer", "predicted_healthy"])

sns.heatmap(confusion, annot=True)
print(classification_report(y_test,y_predict))

#%% To fine tune the parameters of our model, we can play around with the C value for the linear kernel and gamma and C value for RBF:
# I used SVC Linear kernel before. The standard linear SVM has low bias and high variance, but the trade-off can be changed by increasing
#the C parameter that influences the number of violations of the margin allowed in the training data which increases the bias but decreases the variance.


from sklearn.grid_search import GridSearchCV
    
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001],'kernel': ['poly']},
 ]

clf_grid = GridSearchCV(svm.SVC(), param_grid, verbose=1)
 
# Train the classifier
clf_grid.fit(X_train, y_train)
 
# clf = grid.best_estimator_()
print("Best Parameters:\n", clf_grid.best_params_)
print("Best Estimators:\n", clf_grid.best_estimator_)

# The best estimator from the investigation seems to be Linear kernel with a regular C value of 1. Let´s try what is its accuracy.
#%%

C = 1
svc = svm.SVC(kernel='linear', C=C)

svc.fit(X_train_scaled, y_train)


# In[71]:

y_predict= svc.predict(X_test_scaled)


cm= confusion_matrix(y_test,y_predict)

# In[75]:


cm=np.array(confusion_matrix(y_test, y_predict))
confusion= pd.DataFrame(cm, index=["is_cancer", "is_healthy"],columns= ["predicted_cancer", "predicted_healthy"])

sns.heatmap(confusion, annot=True)

print(classification_report(y_test,y_predict))


#%%#%Selecting kernel functions depends on other features of the dataset. 
#If there are a great many features compared to dataset size, you should prefer linear kernels 
#even if the data isn't linearly separable since this reduces the risk of overfitting.
#This was the case in the breast cancer dataset with many factors and high variance.


