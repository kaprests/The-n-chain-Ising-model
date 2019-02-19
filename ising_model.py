import numpy as np
from matplotlib import pyplot as plt
import sympy as sp

B_ex = 2
J_par = 1
J_tan = 1
n_max = 5 # Really n_max -1
beta_const = 0.05

def gen_spin_conf(n):
	spin_conf = np.ones([2**n, n])
	for i in range(n):
		spin_conf.T[i] = np.array([[1]*2**(i) + [-1]*2**(i)] * ((2**n)//(2**(i+1)))).flatten()
	return spin_conf


def P_elem(spin_vec1, spin_vec2, Be, beta, Jp, Jt):
	A = beta*Jp * np.matmul(spin_vec1, spin_vec2)
	B = beta*Jt * np.matmul(spin_vec1, np.roll(spin_vec1, 1))
	C = (beta*Be/2) * np.sum((spin_vec1 + spin_vec2))
	return np.e**(A + B + C)


def gen_P_matrix(P_elem, spin_conf, Be, beta, Jp, Jt):
	n = spin_conf.shape[1]
	P = np.zeros([2**n, 2**n])
	for i in range(2**n):
		for j in range(2**n):
			P[i][j] = P_elem(spin_conf[i], spin_conf[j], Be, beta, Jp, Jt)

	return P


def eigen_values(beta, P_elem, spin_conf, Be, Jp=J_par, Jt=J_tan):
	P = gen_P_matrix(P_elem, spin_conf, Be, beta, Jp, Jt)
	e_vals, e_vecs = np.linalg.eig(P) 
	return np.real(e_vals)


def max_eigen_value(beta, eigen_values, P_elem, spin_conf, Be):
	return np.amax(eigen_values(beta, P_elem, spin_conf, Be))


beta_vals = np.linspace(0.005, 1, 100)
B_vals = np.linspace(-1, 1, 200)
eigen_val_vec = np.zeros([n_max, 100])
eigen_val_vec_b = np.zeros([n_max, 200])
eigen_val_vec_d = np.zeros([n_max, 200])

for i in range(n_max):
	spin_conf = gen_spin_conf(i+2)
	for j in range(beta_vals.size):
		eigen_val_vec[i][j] = np.log(max_eigen_value(beta_vals[j], eigen_values, P_elem, spin_conf, B_ex))
	plt.plot(beta_vals, eigen_val_vec[i])
plt.show()

# The code commented out below is supposed to calculate and plot magnetization, instead it uses a year or so
# to produce a plot of a straight line.
'''
for i in range(n_max):
	spin_conf = gen_spin_conf(i+2)
	for j in range(B_vals.size):
		eigen_val_vec_b[i][j] = np.log(max_eigen_value(beta_const, eigen_values, P_elem, spin_conf, B_vals[j]))
	eigen_val_vec_d[i] = np.gradient(eigen_val_vec_b[i], B_vals)/((i+2)*beta_const)
	plt.plot(B_vals, eigen_val_vec_d[i])
plt.show()

'''
