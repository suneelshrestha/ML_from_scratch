import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split

#load dataset
dataset = load_iris()

#convert to df
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

#standarize data
scalar = StandardScaler()
scaled_df = scalar.fit_transform(df)
scaled_df = pd.DataFrame(df, columns=dataset.feature_names)

#make x and y
x = scaled_df.values
y = dataset.target

#train test
X_train, X_test, Y_tain, Y_test = train_test_split(x, y, test_size=0.2, random_state=23)

knn = neighbors.KNeighborsClassifier(n_neighbors=3)

#fit in the model
knn.fit(X_train,Y_tain)

prediction = knn.predict(X_test)
accuracy = metrics.accuracy_score(Y_test,prediction)

print(accuracy)

for i in range(10):
    print("Actual value:",Y_test[i],"predicted value: ",knn.predict(X_test)[i])
