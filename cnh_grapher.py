import math
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import axes3d
from matplotlib.ticker import LinearLocator

# Real Data
""" df = pd.read_excel("Data_symbolic_regression.xlsx", index_col=None, header=0, usecols="A:C", engine="openpyxl", skiprows= lambda x : x in [0, 0], sheet_name = "Compressible_Neo-Hookean")
x_hat = df["First invariant, X"].values
y_hat = df["Third invariant, Y"].values
w_hat = df["Stored Energy W"].values 

fig = plt.figure()
ax = fig.gca(projection = '3d')
trisurf = ax.plot_trisurf(x_hat, y_hat, w_hat, linewidth=0.2, antialiased=False)
ax.set_xlabel('First invariant, X', fontweight = 'bold')
ax.set_ylabel('Third invariant, Y', fontweight = 'bold')
ax.set_zlabel('Stored Energy W', fontweight = 'bold')
plt.show() """

# Predicted Equation
""" x_pred = np.random.default_rng().uniform(0, 12, 10000)
y_pred = np.random.default_rng().uniform(0, 8, 10000)
w_pred = (y_pred ** 2) * np.log(y_pred) * np.cos((-y_pred + ((np.log(y_pred ** 3) + 1) / y_pred)) / x_pred)

fig = plt.figure()
ax = fig.gca(projection = '3d')
trisurf = ax.plot_trisurf(x_pred, y_pred, w_pred, linewidth=0.2, antialiased=False)
ax.set_xlabel('First invariant, X', fontweight = 'bold')
ax.set_ylabel('Third invariant, Y', fontweight = 'bold')
ax.set_zlabel('Stored Energy W', fontweight = 'bold')
plt.show() """



# y^2 * log(y) * cos((-y + (log(y^3) + 1) / y) / x)