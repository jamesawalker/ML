import csv, numpy as np, matplotlib.pyplot as plt

with open('AB_NYC_2019.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    latitude = []
    longitude = []
    roomtype = []
    price = []
    mindays = [] 
    for row in readCSV:
        # index roomtype
        if row[8]=="Private Room":
            row[8]=0
        else:
            row[8]=1

        latitude.append(row[6])
        longitude.append(row[7])
        roomtype.append(row[8])
        price.append(row[9])
        mindays.append(row[10])
    latitude=np.array(latitude)
    longitude=np.array(longitude)
    roomtype=np.array(roomtype)
    price=np.array(price)
    mindays=np.array(mindays)

colors=(0,0,0)
area = np.pi*3.14
plt.scatter(latitude, longitude, s=area, c=colors, alpha=0.5)
plt.title('Airbnb locations')
plt.show()
