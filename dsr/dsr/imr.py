import numpy as np
import pandas as pd
from dsr import DeepSymbolicRegressor

df = pd.read_excel("Data_symbolic_regression.xlsx", index_col=None, header=0, usecols="A:C", engine="openpyxl", skiprows= lambda x : x in [0, 0])

x_vals = df[["First invariant, X", "Second invariant, Y"]]
y_vals = df["Stored energy W"]

X = x_vals.to_numpy()
y = y_vals.to_numpy()

model = DeepSymbolicRegressor("config.json")

model.fit(X, y)

print(model.program_.pretty())


