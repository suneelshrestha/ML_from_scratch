# implement knn algo using scikit learning algorithm

# import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split


#import data from the sklearn
data = load_breast_cancer()

#convert the data to the dataframe
df = pd.DataFrame(data.data, columns=data.feature_names)

print(df.head())

#now import to the train test split
x = df
y = data.target
scaler = StandardScaler()

x = scaler.fit_transform(x)

# now train the test
X_train,X_test, Y_train ,Y_test = train_test_split(x,y,random_state=23, test_size=0.2)

knn = neighbors.KNeighborsClassifier(n_neighbors=15)

knn.fit(X_train,Y_train)

prediction = knn.predict(X_test)

accuracy = metrics.accuracy_score(Y_test,prediction)
print(accuracy)

for i in range(10):
    print("Actual value:",y[i])
    print("Prediction value:",knn.predict(x)[i])
