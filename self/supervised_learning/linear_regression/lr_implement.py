import numpy as np
import pandas as pd
from lr_from_scratch import LinearRegressionScratch
import matplotlib.pyplot as plt
import math


data = {
    "area":[8,10,12,16,18],
    "price":[18,20,22,24,26]
}
df = pd.DataFrame(data)
X = np.array([1,2,3,4,5,6,7,8])
y = np.array([8,10,12,16,17,20,25,28])

# Create and train the model
reg = LinearRegressionScratch()
reg.fit(X, y)

# Make predictions
predictions = reg.predict(7)
print("Predictions:", predictions)


plt.title("linear Regression")
plt.xlabel("area")
plt.ylabel("price")
plt.scatter(X,y)
plt.plot(X,reg.predict(X), color = "blue")
plt.show()
