
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


sns.pairplot(dataset, hue = "diagnosis", vars = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean"])


# Potential relationship exists therefore I want to investigate it further with the correlation Matrix

# In[23]:


plt.figure(figsize=(20,15))
sns.heatmap (dataset.corr(), annot= True)


# Ther are few strong correlations apart from the more obvious like radius mean and radius worst, more interesting is strong correlation between mean radius and mean perimeter, as well as mean area and mean perimeter

# In[47]:


X= dataset.drop (["diagnosis", "id"], axis=1)


# In[48]:


X.head()


# In[49]:


y = dataset ["diagnosis"]


# In[50]:


y.head()


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)


# In[52]:


X_train.shape


# In[53]:


X_test.shape


# In[54]:


y_train.shape


# In[55]:


y_test.shape


# These are my training and testing data. For the training we have 455 observatios and for testing 114.

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


# # Fitting a SVC model
# Since there are different SVC models to choose from, it is important to recognise the differences between them. I will first try the general linear kerner with a C value of 1. Data is normalised so often a general kernel with a standard parameter C = 1 could sufficle. https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769

# In[69]:


from sklearn import svm, datasets

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
# SVC used (C=1, Linear Kernel...more model details are on line 70) gives reasonable results and based on the confusion matrix, it only predicts one time that the tumor is cancerous while it was actually healthy. 
#More concerning is the fact that it predicted 11 times that the tumor is healthy while in fact it was cancerous. All in all it give around 90% accuracy.

# # We can try to improve model accuracy with tweaking model parameters.
# I used SVC Linear kernel before. Now I will try to use more sophisticated  modedl which can map it to a higher dimensional space like a Gaussian kernel. 
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

# Interestingly the unscaled dataset performed better in terms of accuracy

#%% I will try a more complex Gaussian Kernel

C = 10.0
svclassifier = svm.SVC(kernel='rbf',C=C)  
svclassifier.fit(X_train_scaled, y_train)  

y_predict = svclassifier.predict(X_test_scaled)  


# In[86]:


cm= confusion_matrix(y_test,y_predict)
cm=np.array(confusion_matrix(y_test, y_predict))
confusion= pd.DataFrame(cm, index=["is_cancer", "is_healthy"],columns= ["predicted_cancer", "predicted_healthy"])
confusion
sns.heatmap(confusion, annot=True)
print(classification_report(y_test,y_predict))

#%% The Gaussian kernel performed slightly better than the linear kernel on scaled data. Let´s see how it performs on the unscaled dataset.

C = 1
svclassifier = svm.SVC(kernel='rbf',C=C)  
svclassifier.fit(X_train, y_train)  

y_predict = svclassifier.predict(X_test)  

cm= confusion_matrix(y_test,y_predict)
cm=np.array(confusion_matrix(y_test, y_predict))
confusion= pd.DataFrame(cm, index=["is_cancer", "is_healthy"],columns= ["predicted_cancer", "predicted_healthy"])
confusion
sns.heatmap(confusion, annot=True)
print(classification_report(y_test,y_predict))

#%%The gamma parameter is used for the Gaussian kernel function. The kernel functions can be seen as an efficient way to transform your original features into another space,
#where a separating hyperplane in the new feature space does not have to be linear in the original feature space.
#So I will try to adjust the gamma parameter of the Gaussian fucntion and see it it works better.




# # Plot different SVM classifiers 
# Comparison of different linear SVM classifiers on a 2D projection of the iris dataset. We only consider the first 2 features of this dataset:
# 
# Sepal length
# Sepal width
# This example shows how to plot the decision surface for four SVM classifiers with different kernels.
# 
# The linear models LinearSVC() and SVC(kernel='linear') yield slightly different decision boundaries. This can be a consequence of the following differences:
# 
# LinearSVC minimizes the squared hinge loss while SVC minimizes the regular hinge loss.
# 
# LinearSVC uses the One-vs-All (also known as One-vs-Rest) multiclass reduction while SVC uses the One-vs-One multiclass reduction.
# 
# Both linear models have linear decision boundaries (intersecting hyperplanes) while the non-linear kernel models (polynomial or Gaussian RBF) have more flexible non-linear decision boundaries with shapes that depend on the kind of kernel and its parameters.
# 
# .. NOTE:: while plotting the decision function of classifiers for toy 2D datasets can help get an intuitive understanding of their respective expressive power, be aware that those intuitions don't always generalize to more realistic high-dimensional problems.

# In[102]:
#
#
#def make_meshgrid(x, y, h=.02):
#    """Create a mesh of points to plot in
#
#    Parameters
#    ----------
#    x: data to base x-axis meshgrid on
#    y: data to base y-axis meshgrid on
#    h: stepsize for meshgrid, optional
#
#    Returns
#    -------
#    xx, yy : ndarray
#    """
#    x_min, x_max = x.min() - 1, x.max() + 1
#    y_min, y_max = y.min() - 1, y.max() + 1
#    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                         np.arange(y_min, y_max, h))
#    return xx, yy
#
#
#def plot_contours(ax, clf, xx, yy, **params):
#    """Plot the decision boundaries for a classifier.
#
#    Parameters
#    ----------
#    ax: matplotlib axes object
#    clf: a classifier
#    xx: meshgrid ndarray
#    yy: meshgrid ndarray
#    params: dictionary of params to pass to contourf, optional
#    """
#    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#    Z = Z.reshape(xx.shape)
#    out = ax.contourf(xx, yy, Z, **params)
#    return out
#
#X=X_train.iloc[:, [1]]
#Y=y_train.iloc[:, [0]]
#
#
## we create an instance of SVM and fit out data. We do not scale our
## data since we want to plot the support vectors
#C = 1.0  # SVM regularization parameter
#models = (svm.SVC(kernel='linear', C=C),
#          svm.LinearSVC(C=C),
#          svm.SVC(kernel='rbf', gamma=0.7, C=C),
#          svm.SVC(kernel='poly', degree=3, C=C))
#models = (clf.fit(X, y) for clf in models)
#
## title for the plots
#titles = ('SVC with linear kernel',
#          'LinearSVC (linear kernel)',
#          'SVC with RBF kernel',
#          'SVC with polynomial (degree 3) kernel')
#
## Set-up 2x2 grid for plotting.
#fig, sub = plt.subplots(2, 2)
#plt.subplots_adjust(wspace=0.4, hspace=0.4)
#
#X0, X1 = X[:, 0], X[:, 1]
#xx, yy = make_meshgrid(X0, X1)
#
#for clf, title, ax in zip(models, titles, sub.flatten()):
#    plot_contours(ax, clf, xx, yy,
#                  cmap=plt.cm.coolwarm, alpha=0.8)
#    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
#    ax.set_xlim(xx.min(), xx.max())
#    ax.set_ylim(yy.min(), yy.max())
#    ax.set_xlabel('radius_mean')
#    ax.set_ylabel('texture_mean')
#    ax.set_xticks(())
#    ax.set_yticks(())
#%% To fine tune the parameters of our model, we can play around with the C value for the linear kernela and gamma and C value for RBF:
# I used SVC Linear kernel before. The standard linear SVM has low bias and high variance, but the trade-off can be changed by increasing
#the C parameter that influences the number of violations of the margin allowed in the training data which increases the bias but decreases the variance.


from sklearn.grid_search import GridSearchCV
    
param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]

clf_grid = GridSearchCV(svm.SVC(), param_grid, verbose=1)
 
# Train the classifier
clf_grid.fit(X_train, y_train)
 
# clf = grid.best_estimator_()
print("Best Parameters:\n", clf_grid.best_params_)
print("Best Estimators:\n", clf_grid.best_estimator_)

# The best estimator from the investigation seems to be Linear kernel with a large C value of 1000. Let´s try what is its accuracy.
#%%

C = 100000
svc = svm.SVC(kernel='linear', C=C)

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

#%%

from sklearn.grid_search import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
# Build a classification task using 3 informative features
X_train, y_train = make_classification(n_samples=1000,
                           n_features=10,
                           n_informative=3,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)


rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True) 

param_grid = { 
    'n_estimators': [200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(X, y)
print (CV_rfc.best_params_)

#%% by the way, selecting kernel functions depends on other features of the dataset. 
#If there are a great many features compared to dataset size, you should prefer linear kernels 
#even if the data isn't linearly separable since this reduces the risk of overfitting.
