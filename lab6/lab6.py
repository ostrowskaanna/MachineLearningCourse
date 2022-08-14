#!/usr/bin/env python
# coding: utf-8

# In[297]:


from sklearn import datasets
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)


# In[298]:


X = data_breast_cancer.data[['mean texture', 'mean symmetry']]
y = data_breast_cancer.target


# In[299]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[300]:


from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier().fit(X_train, y_train)


# In[301]:


from sklearn.linear_model import LogisticRegression 
log_clf = LogisticRegression().fit(X_train, y_train)


# In[302]:


from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier().fit(X_train, y_train)


# In[303]:


from sklearn.ensemble import VotingClassifier
vot_clf_hard = VotingClassifier(estimators=[('tc', tree_clf),('lr', log_clf),('knc', knn_clf)], voting='hard').fit(X_train, y_train)
vot_clf_soft = VotingClassifier(estimators=[('tc', tree_clf),('lr', log_clf),('knc', knn_clf)], voting='hard').fit(X_train, y_train)


# In[304]:


from sklearn.metrics import accuracy_score 
result = []
tree_acc_train = accuracy_score(y_train, tree_clf.predict(X_train))
tree_acc_test = accuracy_score(y_test, tree_clf.predict(X_test))
acc = (tree_acc_train, tree_acc_test)
result.append(acc)
log_acc_train = accuracy_score(y_train, log_clf.predict(X_train))
log_acc_test = accuracy_score(y_test, log_clf.predict(X_test))
acc = (log_acc_train, log_acc_test)
result.append(acc)
knn_acc_train = accuracy_score(y_train, knn_clf.predict(X_train))
knn_acc_test = accuracy_score(y_test, knn_clf.predict(X_test))
acc = (knn_acc_train, knn_acc_test)
result.append(acc)
vot_hard_acc_train = accuracy_score(y_train, vot_clf_hard.predict(X_train))
vot_hard_acc_test = accuracy_score(y_test, vot_clf_hard.predict(X_test))
acc = (vot_hard_acc_train, vot_hard_acc_test)
result.append(acc)
vot_soft_acc_train = accuracy_score(y_train, vot_clf_soft.predict(X_train))
vot_soft_acc_test = accuracy_score(y_test, vot_clf_soft.predict(X_test))
acc = (vot_soft_acc_train, vot_soft_acc_test)
result.append(acc)


# In[305]:


result


# In[306]:


import pickle
with open('acc_vote.pkl', 'wb') as file:
    pickle.dump(result, file)


# In[307]:


classifiers = [tree_clf, log_clf, knn_clf, vot_clf_hard, vot_clf_soft]


# In[308]:


with open('vote.pkl', 'wb') as file:
    pickle.dump(classifiers, file)


# In[309]:


from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30).fit(X_train, y_train)
bag_clf_2 = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, max_samples=0.5).fit(X_train, y_train)
past_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, bootstrap=False).fit(X_train, y_train)
past_clf_2 = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, max_samples=0.5, bootstrap=False).fit(X_train, y_train)
rnd_clf = RandomForestClassifier(n_estimators=30).fit(X_train ,y_train)
ada_clf = AdaBoostClassifier(n_estimators=30).fit(X_train, y_train)
grad_clf = GradientBoostingClassifier(n_estimators=30).fit(X_train, y_train)


# In[310]:


result = []
bag_acc_train = accuracy_score(y_train, bag_clf.predict(X_train))
bag_acc_test = accuracy_score(y_test, bag_clf.predict(X_test))
acc = (bag_acc_train, bag_acc_test)
result.append(acc)
bag_2_acc_train = accuracy_score(y_train, bag_clf_2.predict(X_train))
bag_2_acc_test = accuracy_score(y_test, bag_clf_2.predict(X_test))
acc = (bag_2_acc_train, bag_2_acc_test)
result.append(acc)
past_acc_train = accuracy_score(y_train, past_clf.predict(X_train))
past_acc_test = accuracy_score(y_test, past_clf.predict(X_test))
acc = (past_acc_train, past_acc_test)
result.append(acc)
past_2_acc_train = accuracy_score(y_train, past_clf_2.predict(X_train))
past_2_acc_test = accuracy_score(y_test, past_clf_2.predict(X_test))
acc = (past_2_acc_train, past_2_acc_test)
result.append(acc)
rnd_acc_train = accuracy_score(y_train, rnd_clf.predict(X_train))
rnd_acc_test = accuracy_score(y_test, rnd_clf.predict(X_test))
acc = (rnd_acc_train, rnd_acc_test)
result.append(acc)
ada_acc_train = accuracy_score(y_train, ada_clf.predict(X_train))
ada_acc_test = accuracy_score(y_test, ada_clf.predict(X_test))
acc = (ada_acc_train, ada_acc_test)
result.append(acc)
grad_acc_train = accuracy_score(y_train, grad_clf.predict(X_train))
grad_acc_test = accuracy_score(y_test, grad_clf.predict(X_test))
acc = (grad_acc_train, grad_acc_test)
result.append(acc)


# In[311]:


result


# In[312]:


with open('acc_bag.pkl', 'wb') as file:
    pickle.dump(result, file)


# In[313]:


classifiers = [bag_clf, bag_clf_2, past_clf, past_clf_2, rnd_clf, ada_clf, grad_clf]


# In[314]:


with open('bag.pkl', 'wb') as file:
    pickle.dump(classifiers, file)


# In[315]:


X = data_breast_cancer.data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
sampling = BaggingClassifier(DecisionTreeClassifier(), n_estimators=30, 
                             max_features=2, bootstrap=True, bootstrap_features=False).fit(X_train, y_train)


# In[316]:


acc = [accuracy_score(y_train, sampling.predict(X_train)), accuracy_score(y_test, sampling.predict(X_test))]
print(acc)
clf = [sampling]
print(clf)


# In[317]:


with open('acc_fea.pkl', 'wb') as file:
    pickle.dump(acc, file)
with open('fea.pkl', 'wb') as file:
    pickle.dump(clf, file)


# In[318]:


train_acc = []
test_acc = []
feat_names = []
for i in range(0, 30):
    clf = sampling.estimators_[i]
    feature1 = data_breast_cancer.feature_names[sampling.estimators_features_[i][0]]
    feature2 = data_breast_cancer.feature_names[sampling.estimators_features_[i][1]]
    X_ = data_breast_cancer.data[[feature1, feature2]]
    X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.2)
    train_acc.append(accuracy_score(y_train, clf.predict(X_train)))
    test_acc.append(accuracy_score(y_test, clf.predict(X_test)))
    feat_names.append([feature1, feature2])


# In[319]:


import pandas as pd
d = {'train accuracy': train_acc, 'test accuracy': test_acc, 'feature names': feat_names}    
df = pd.DataFrame(data=d)
df.sort_values(by=['train accuracy', 'test accuracy'], ascending=False)


# In[320]:


with open('acc_fea_rank.pkl', 'wb') as file:
    pickle.dump(df, file)


# In[ ]:




