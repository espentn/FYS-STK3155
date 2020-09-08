import numpy as np

class Regression:
	
	def __init__(self, poly_order):
		self.n = poly_order

	def createDesignMatrix(self,x,y):
		if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

		N = len(x)
		p = int((self.n+1)*(self.n+2)/2)
		X = np.ones((N,p))

		for i in range(1, n+1):
			q = int(i*(i+1)/2)
			for j in range(i+1):
				X[:,q+j] = (x**(i-j))*(y**j)
		return X

	def MSE(self, z_data, z_model):
		n = np.size(z_model)
		return np.sum((z_data-z_model)**2)/n

	def R2(z_data, z_model):
		n = np.size(z_data)
		return 1 - np.sum((z_data-z_model)**2)/np.sum((z_data-(np.sum(z_data)/n))**2)

	def relativeError(self, z_data, z_model):
		return abs((z_data - z_model)/z_data)

