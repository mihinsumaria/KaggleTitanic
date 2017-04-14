#Clusters(Age and Fare)=3,2, 0.77033
#Clusters(Age and Fare)=4,3, 0.77512
#Clusters(Age and Fare)=5,3, 0.76555
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

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
X=data['Age'].values.reshape(-1,1)
Y=data['Fare'].values.reshape(-1,1)

n=int(raw_input('Enter number of clusters for age: '))
m=int(raw_input('Enter number of clusters for fare: '))
kmeans=KMeans(n_clusters=n,random_state=0).fit(X)
kmeans1=KMeans(n_clusters=m,random_state=0).fit(Y)
#colors=cm.rainbow(np.linspace(0,1,4))
#colors1=cm.rainbow(np.linspace(0,1,4))
data['AnL']=kmeans.labels_
data['FmL']=kmeans1.labels_

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

X=test['Age'].values.reshape(-1,1)
Y=test['Fare'].values.reshape(-1,1)
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

print "Running Random Forest Classifier"
rfc=RandomForestClassifier()
testpred=rfc.fit(training,target).predict(testing)
test['Survived']=testpred

test[['PassengerID','Survived']].to_csv('RandomForest.csv',index=False)