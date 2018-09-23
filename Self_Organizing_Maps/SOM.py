import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone,pcolor,colorbar,plot,show

# Using Data from UCI Machine Learning Repository on Australian Credit Approval
# http://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval)
'''
Using Self Organizing Maps to detect fraud in Credit Card Applications using Australian Credit Approval Dataset
'''
dataset = pd.read_csv("Credit_Card_Applications.csv")

x = dataset.iloc[:, :-1].values  # Everything except final row
y = dataset.iloc[:, -1].values  # Final Row

# Data isn't between 0 and 1 so we need MinMaxScaler again from sci-kit learn

scaler = MinMaxScaler(feature_range=(0, 1))
x_scaled = scaler.fit_transform(x) # Scales every column to be in between 0 and 1 through normalization

# For SOM someone already created an architecture for SOM's, using MiniSOM 1.0, license CC BY 3.0
# Training SOM
SOM = MiniSom(x=10, y=10, input_len=15)
SOM.random_weights_init(x_scaled)  # Initialize Weights
SOM.train_random(data=x_scaled, num_iteration=100)

bone()
pcolor(SOM.distance_map())
colorbar()
markers = ['o','s']
colors = ['r','g']
for i, x in enumerate(x_scaled):
    w = SOM.winner(x)
    plot(w[0]+0.5, w[1]+0.5, markers[y[i]], markeredgecolor=colors[y[i]], markerfacecolor='None', markersize=10, markeredgewidth=2)
show()

# Finding Anomalies
mappings = SOM.win_map(x_scaled)
# Change to not be hardcoded
frauds = np.concatenate((mappings[(8, 1)], mappings[(6, 8)]),axis=0)
frauds = scaler.inverse_transform(frauds)


