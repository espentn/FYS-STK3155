import numpy as np
from Regression import createDesignMatrix

class OrdinaryLeastSquares:

	def fitBeta(self,X, z_data):
		# Using the SVD algorithm to fit beta
		U, s, VT = np.linalg.svd(X)

		D = np.zeros((len(U),len(VT)))
		for i in range(0,len(VT)):
			D[i,i] = s[i]

		invD = np.linalg.inv(D)
		UT = U.T
		V = VT.T

		X = np.matmul(V,np.matmul(invD,UT))

		beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z_data)
		return beta