from dsr import DeepSymbolicRegressor
import numpy as np

# Generate some data
np.random.seed(721)
X = np.random.random((1000, 1))
y = np.sin(X[:,0] ** 2) * np.cos(X[:,0]) - 1

# Create the model
model = DeepSymbolicRegressor("config.json")

# Fit the model
model.fit_v2(X, y) 

# View the best expression
print(model.program_.pretty())

# Make predictions
#model.predict(2 * X)



""" data = {'col1' : np.arange(0, 1000), 'col2' : np.arange(0, 1000)}

df_xy = pd.DataFrame(data=data)
xy = df_xy.to_numpy() 
print(xy)

z = []

for x, y in xy:
    val = (x**4) - (x**3) + (0.5 * (y**2)) + y
    z.append(val)

print(z) """

""" model = DeepSymbolicRegressor("config.json")
model.fit(xa, ya) """
#print(model.program_.pretty())


# r(x), n(x)
# r(x) - n(x) = e(x)
#  n(x) + e(x) or e(n(x))