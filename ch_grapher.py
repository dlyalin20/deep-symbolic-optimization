import math
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import axes3d
from matplotlib.ticker import LinearLocator

# Real Data
""" df = pd.read_excel("Data_symbolic_regression.xlsx", index_col=None, header=0, usecols="A:C", engine="openpyxl", skiprows= lambda x : x in [0, 0], sheet_name = "Clay_hyperelasticity")
x_hat = df["Volumetric invariant, X"].values
y_hat = df["Deviatoric invariant, Y"].values
w_hat = df["Stored energy W"].values 

fig = plt.figure()
ax = fig.gca(projection = '3d')
trisurf = ax.plot_trisurf(x_hat, y_hat, w_hat, linewidth=0.2, antialiased=False)
ax.set_xlabel('Volumetric invariant, X', fontweight = 'bold')
ax.set_ylabel('Deviatoric invariant, Y', fontweight = 'bold')
ax.set_zlabel('Stored energy W', fontweight = 'bold')
plt.show() """

# Predicted Equation
""" x_pred = np.random.default_rng().uniform(-0.3, 0.3, 10000)
y_pred = np.random.default_rng().uniform(0, 2, 10000)
w_pred = np.e ** ((-2 * x_pred) - (2 * y_pred) * np.sin(y_pred) + (3 * y_pred) + np.e ** (np.sin((x_pred ** 2) * np.e ** y_pred)))

fig = plt.figure()
ax = fig.gca(projection = '3d')
trisurf = ax.plot_trisurf(x_pred, y_pred, w_pred, linewidth=0.2, antialiased=False)
ax.set_xlabel('Volumetric invariant, X', fontweight = 'bold')
ax.set_ylabel('Deviatoric invariant, Y', fontweight = 'bold')
ax.set_zlabel('Stored energy W', fontweight = 'bold')
plt.show() """

# e ^ (-2x - 2ysin(y) + 3y + e^sin(x^2 * e^y))