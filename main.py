from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
from scikot-learn import train_test_split
from sklearn.preprocessing import StandardScaler

fig = plt.figure()
ax = fig.gca(projection='3d')


def frankeFunction(x,y):
	#noise = np.random.normal(0.5,1,len(x))
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4 #+ noise

def createDataPoints(x,y):
	x_d, y_d = np.meshgrid(x,y)
	z_d = FrankeFunction(x_d,y_d)
	return x_d, y_d, z_d

# Need to flatten the matrices to be able to compute the beta
def convertDataPoints(x,y,z):
	x_d = np.ravel(x)
	y_d = np.ravel(y)
	z_d = np.ravel(z)
	return x_d, y_d, z_d

def create_design_matrix(x, y):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	n = len(x)
	p = int((n+1)*(n+2)/2)
	X = np.ones((N,p))

	for i in range(1, n+1):
		q = int(i*(i+1)/2)
		for j in range(i+1):
			X[:,q+j] = (x**(i-j))*(y**j)
	return X


def predict(X, z_data):
	beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z_data)
	ztilde = X @ beta
    return ztilde

def splitData(x,z):
    X_train, X_test, z_train, z_test = train_test_split(x,z,test_size=0.3)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    Beta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(z_train)
    ztilde = X_train @ Beta
    return ztilde

def MSE(z_data, z_model):
	n = np.size(z_model)
	return np.sum((z_data-z_model)**2)/n

def R2(z_data, z_model):
	n = np.size(z_data)
	return 1 - np.sum((z_data-z_model)**2)/np.sum((z_data-(np.sum(z_data)/n))**2)



# Make data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
noise = np.random.rand()

x, y = np.meshgrid(x,y)
z = frankeFunction(x, y)

# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()