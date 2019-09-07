import numpy as np, matplotlib.pyplot as plt, pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#data set cleaned in earlier step, removed misaligned rows and price>1000
d = pd.read_csv("AB_NYC_2019.csv")

latitude = d["latitude"]
longitude = d["longitude"]
price = d["price"]
roomtype = d["room_type"]
minstay = d["minimum_nights"]

#reformat the data into an array length n_samples, width n_frames, split into training and test
lat = np.array(latitude)
lon = np.array(longitude)
pri = np.array(price)
roo = np.array(roomtype)
mis = np.array(minstay)

X=np.array([lat,lon,roo,mis])
X=X.transpose()

X_train, X_test, pri_train, pri_test = train_test_split(X,pri)

# scale the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPRegressor(hidden_layer_sizes=(20,20,20),max_iter=1500)
mlp.fit(X_train,pri_train)
predictions=mlp.predict(X_test)

plt.hist2d(predictions,pri_test,100,[[0,400],[0,400]])
print np.corrcoef(predictions,pri_test)

plt.show()
#plt.hist(latitude,bins=100)
#plt.show()
