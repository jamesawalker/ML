import numpy as np, matplotlib.pyplot as plt, pandas as pd
from sklearn.linear_model import Perceptron

d = pd.read_csv("AB_NYC_2019.csv")

latitude = d["latitude"]
longitude = d["longitude"]
price = d["price"]
roomtype = d["room_type"]
minstay = d["minimum_nights"]

#reformat the data into an array length n_samples, width n_frames, and cut out the last ~8000 for verification set
lat = np.array(latitude[0:40000])
lon = np.array(longitude[0:40000])
pri = np.array(price[0:40000])
roo = np.array(roomtype[0:40000])
mis = np.array(minstay[0:40000])

X=np.array([lat,lon,roo,mis])
X=X.transpose()

#plt.hist(latitude,bins=100)
#plt.show()
