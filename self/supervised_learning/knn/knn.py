import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#load the brest cancer data
dataset = load_breast_cancer()

#create a data frame
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

#preprocessing the dataframe
scalar = StandardScaler()
scaled_df = scalar.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df,columns=dataset.feature_names)

#divide the features as x and target as y
x = scaled_df.values
y = dataset.target

#devide train test the model with 20% test and 80% train
X_train,X_test,Y_train,Y_test = train_test_split(x, y, test_size=0.2, random_state=23)

#now prediction
knn = neighbors.KNeighborsClassifier(n_neighbors=3)

#now train the model
knn.fit(X_train,Y_train)

#make prediction
prediction = knn.predict(X_test)

accuracy = metrics.accuracy_score(Y_test,prediction)
print(accuracy)

for i in range(10):
    print("actual result:",Y_test[i]," predicted : ",knn.predict(X_test)[i])
