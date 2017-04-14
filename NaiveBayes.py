"""
Clusters (Age) = 4, Clusters(Fare) = 3, 0.70335.
Clusters (Age) = 3, Clusters(Fare) = 2, 0.74641.
Clusters (Age) = 5, Clusters(Fare) = 3, 0.71292.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.cm as cm
from sklearn.metrics import silhouette_score
from sklearn.naive_bayes import GaussianNB

data=pd.read_csv('data/train_pandas.csv',names=['Passenger','Survived','PClass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])

#Estimating missing age values using PClass and Sex for training data
print "Estimating missing age values for training data"
for j in range(1,4):
	mean=data[(data['Sex']=='male') & (data['PClass']==j)]['Age'].mean()
	for i in data[(data['Sex']=='male') & (data['PClass']==j)]['Age'].index:
		if(np.any(np.isnan(data.loc[i,'Age']))):
			data.loc[i,'Age']=mean
for j in range(1,4):
	mean=data[(data['Sex']=='female') & (data['PClass']==j)]['Age'].mean()
	for i in data[(data['Sex']=='female') & (data['PClass']==j)]['Age'].index:
		if(np.any(np.isnan(data.loc[i,'Age']))):
			data.loc[i,'Age']=mean

print "Clustering Age and Fare from training data"
X=data['Age'].reshape(-1,1)
Y=data['Fare'].reshape(-1,1)

n=int(raw_input('Enter number of clusters for age: '))
m=int(raw_input('Enter number of clusters for fare: '))
kmeans=KMeans(n_clusters=n,random_state=0).fit(X)
kmeans1=KMeans(n_clusters=m,random_state=0).fit(Y)
#colors=cm.rainbow(np.linspace(0,1,4))
#colors1=cm.rainbow(np.linspace(0,1,4))
data['AnL']=kmeans.labels_
data['FmL']=kmeans1.labels_

"""
print "Plotting clusters"
plt.figure(1)

plt.subplot(211)
for i in range(4):
	dA4L=data[data['A4L']==i]
	X=dA4L['Age']
	plt.scatter(X,np.zeros(len(X)),color=colors[i])

plt.subplot(212)
for i in range(4):
	dF4L=data[data['F4L']==i]
	Y=dF4L['Fare']
	plt.scatter(Y,np.zeros(len(Y)),color=colors1[i])
"""

"""
We compute silhouette scores for k ranging from 3 to 10 for both Age and Fare
Based on the analysis, we go for k=4 for both of them

si=[]
sj=[]
for i in range(3,10):
	kmeansi=KMeans(n_clusters=i,random_state=0)
	kmeansj=KMeans(n_clusters=i,random_state=0)
	kmeansi.fit(X)
	kmeansj.fit(Y)
	labelsi=kmeansi.labels_
	centroidsi=kmeansi.cluster_centers_
	labelsj=kmeansj.labels_
	centroidsj=kmeansj.cluster_centers_
	si.append(silhouette_score(X,labelsi,metric='euclidean'))
	sj.append(silhouette_score(Y,labelsj,metric='euclidean'))

plt.figure(1)

plt.subplot(211)
plt.plot(si)
plt.ylabel("Silouette")
plt.xlabel("k")
plt.title("Silouette for K-means cell's behaviour (Age)")

plt.subplot(212)
plt.plot(sj)
plt.ylabel("Silouette")
plt.xlabel("k")
plt.title("Silouette for K-means cell's behaviour (Fare)")

plt.show()
"""
print "Replacing with categorical values (Cabin-Train)"
data['Cabin'].fillna(value=0,inplace=True)
data['Cabin'].replace(to_replace='\D*',value=1,inplace=True,regex=True)

print "Replacing with categorical values (Embarked-Train)"
data['Embarked'].replace(to_replace='S',value=1,inplace=True)
data['Embarked'].replace(to_replace='Q',value=2,inplace=True)
data['Embarked'].replace(to_replace='C',value=3,inplace=True)

print "Replacing with categorical values (Sex-Train)"
data['Sex'].replace(to_replace='male',value=1,inplace=True)
data['Sex'].replace(to_replace='female',value=1,inplace=True)

training=data[['PClass','Sex','AnL','FmL','Cabin','Embarked']]
target=data['Survived']

#importing test data
print "Importing testing data"
test=pd.read_csv('data/test_pandas.csv',names=['PassengerID','PClass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])

#Estimating missing age values using PClass and Sex for testing data
print "Estimating missing age values for testing data"
for j in range(1,4):
	mean=test[(test['Sex']=='male') & (test['PClass']==j)]['Age'].mean()
	for i in test[(test['Sex']=='male') & (test['PClass']==j)]['Age'].index:
		if(np.any(np.isnan(test.loc[i,'Age']))):
			test.loc[i,'Age']=mean
for j in range(1,4):
	mean=test[(test['Sex']=='female') & (test['PClass']==j)]['Age'].mean()
	for i in test[(test['Sex']=='female') & (test['PClass']==j)]['Age'].index:
		if(np.any(np.isnan(test.loc[i,'Age']))):
			test.loc[i,'Age']=mean

X=test['Age'].reshape(-1,1)
Y=test['Fare'].reshape(-1,1)
#Check for missing values
#print np.isnan(X).any()
#print np.isnan(Y).any()

print "Predicting cluster labels for test data"
test['AnL']=kmeans.predict(X)
test['FmL']=kmeans1.predict(Y)

print "Replacing with categorical values (Cabin-Test)"
test['Cabin'].fillna(value=0,inplace=True)
test['Cabin'].replace(to_replace='\D*',value=1,inplace=True,regex=True)

print "Replacing with categorical values (Embarked-Test)"
test['Embarked'].replace(to_replace='S',value=1,inplace=True)
test['Embarked'].replace(to_replace='Q',value=2,inplace=True)
test['Embarked'].replace(to_replace='C',value=3,inplace=True)

print "Replacing with categorical values (Sex-Test)"
test['Sex'].replace(to_replace='male',value=1,inplace=True)
test['Sex'].replace(to_replace='female',value=1,inplace=True)

testing=test[['PClass','Sex','AnL','FmL','Cabin','Embarked']]
"""
#print np.any(np.isnan(training['PClass']))
#print np.any(np.isnan(training['Sex']))
#print np.any(np.isnan(training['A4L']))
#print np.any(np.isnan(training['F4L']))
#print np.any(np.isnan(training['Cabin']))
#print np.any(np.isnan(training['Embarked']))

"""
print "Running Gaussian Naive Bayes"
gnb=GaussianNB()
testpred=gnb.fit(training,target).predict(testing)
test['Survived']=testpred

test[['PassengerID','Survived']].to_csv('NaiveBayes.csv',index=False)