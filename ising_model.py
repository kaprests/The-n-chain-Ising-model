import numpy as np
from matplotlib import pyplot as plt
import sympy as sp

B_ex = 2
J_par = 1
J_tan = 1
n_max = 2 # Really n_max -1
beta_const = 0.5

def gen_spin_conf(n):
	spin_conf = np.ones([2**n, n])
	for i in range(n):
		spin_conf.T[i] = np.array([[1]*2**(i) + [-1]*2**(i)] * ((2**n)//(2**(i+1)))).flatten()
	return spin_conf


def P_elem(spin_vec1, spin_vec2, Be, beta, Jp, Jt):
	A = beta*Jp * np.matmul(spin_vec1, spin_vec2)
	B = beta*Jt * np.matmul(spin_vec1, np.roll(spin_vec1, 1))
	C = (beta*Be/2) * np.sum((spin_vec1 + spin_vec2))
	return A, B, C
	#return np.exp(A + B + C)


def gen_P_matrix(P_elem, spin_conf, Be, beta, Jp, Jt):
	n = spin_conf.shape[1]
	P = np.zeros([2**n, 2**n])
	A, B, C = np.zeros([2**n, 2**n]), np.zeros([2**n, 2**n]), np.zeros([2**n, 2**n])
	for i in range(2**n):
		for j in range(2**n):
			#P[i][j] = P_elem(spin_conf[i], spin_conf[j], Be, beta, Jp, Jt)
			A[i][j], B[i][j], C[i][j] = P_elem(spin_conf[i], spin_conf[j], Be, beta, Jp, Jt)

	return A, B, C
#	return P

# P_test tries to compute A, B and C matrices to compare with the matrices from P_fast
# P_test and P_fast however returns the same as each other, they are both wrong.
# Maybe easier to debug P_test, and then apply corresponding corrections to P_fast

def P_test(spin_conf, Be, beta, Jp, Jt):
	n = spin_conf.shape[1]
	A = np.zeros([2**n, 2**n])
	B = np.zeros([2**n, 2**n])
	C = np.zeros([2**n, 2**n])
	
	for i in range(2**n):
		for j in range(2**n):
			A[i][j] = beta*Jp * np.matmul(spin_conf[i], spin_conf[j])
			B[i][j] = beta*Jt * np.matmul(spin_conf[i], np.roll(spin_conf[j], 1))
			C[i][j] = (beta*Be/2) * np.sum((spin_conf[i] + spin_conf[j]))
	return A, B, C


def P_fast(spin_conf, Be, beta, Jp, Jt):
	A = beta*Jp * np.matmul(spin_conf, spin_conf.T)
	B = beta*Jt * np.matmul(spin_conf, np.apply_along_axis(np.roll, 1, spin_conf, 1).T)
	sum_vec = np.sum(spin_conf, axis=1)
	C = (beta*Be/2) * np.repeat(sum_vec.reshape(len(sum_vec), 1), len(sum_vec), axis=1) + sum_vec
	# print(np.repeat(sum_vec.reshape(len(sum_vec), 1), len(sum_vec), axis=1) + sum_vec) 
	#print(A, "A inne i fast")
	#print(B, "B inne i fast")
	#print(C, "C inne i fast")
	return A, B, C
	#return np.exp(A + B + C)

spin_conf = gen_spin_conf(3)
A, B, C = gen_P_matrix(P_elem, spin_conf, 1, 1, 1, 1)
A2, B2, C2 = P_fast(spin_conf, 1, 1, 1, 1)
A3, B3, C3 = P_test(spin_conf, 1, 1, 1, 1)

print("P_fast")
print(A2)
print(B2)
print(C2)

print("P_test")
print(A3)
print(B3)
print(C3)

print("P_right")
print(A, "A rett")
print(B, "B rett")
print(C, "C rett")
'''
spin_conf = gen_spin_conf(2)
A, B, C = P_test(spin_conf, 1, 2, 2, 1)
A2, B2, C2, = P_fast(spin_conf, 1, 2, 2, 1)
P1 = gen_P_matrix(P_elem, spin_conf, 1, 1, 1, 1)
P2 = np.exp(A2 + B2 + C2)
P3 = np.exp(A + B + C)

print(P1)
print(" ")
print(P2)
print(" ")
print(P3)

#print(A - A2, "\n\n")
#print(B - B2, "\n\n")
#print(C - C2, "\n\n")
'''

'''

def max_eigen_value(beta, P_elem, spin_conf, Be, Jp=J_par, Jt=J_tan):
	P = gen_P_matrix(P_elem, spin_conf, Be, beta, Jp, Jt)
	e_vals, e_vecs = np.linalg.eig(P) 
	return np.amax(np.real(e_vals))


def max_eigen_value_fast(beta, P_fast, spin_conf, Be, Jp=J_par, Jt=J_tan):
	e_vals, e_vecs = np.linalg.eig(P_fast(spin_conf, Be, beta, Jp, Jt))
	return np.amax(np.real(e_vals))


beta_vals = np.linspace(0.005, 1, 100)
B_vals = np.linspace(-1, 1, 100)
eigen_val_vec = np.zeros([n_max, 100])
eigen_val_vec_b = np.zeros([n_max, 100])
eigen_val_vec_d = np.zeros([n_max, 100])
spes_heat_vec = np.zeros([5, 100])
'''

'''
for i in range(n_max):
	spin_conf = gen_spin_conf(i+2)
	print(i+2)
	for j in range(beta_vals.size):
		eigen_val_vec[i][j] = np.log(max_eigen_value(beta_vals[j], P_elem, spin_conf, B_ex))
		eigen_val_vec_b[i][j] = np.log(max_eigen_value(beta_const, P_elem, spin_conf, B_vals[j]))
	if i+2 <= 5:
		for k in range(beta_vals.size):
			spes_heat_vec[i] = np.gradient(np.gradient(eigen_val_vec[i], beta_vals), beta_vals)*(beta_vals[k]/(i+2))
	eigen_val_vec_d[i] = np.gradient(eigen_val_vec_b[i], B_vals)/((i+2)*beta_const)
	plt.plot(beta_vals, eigen_val_vec[i])
	print(i+1," done", n_max - i -1, " to go.")
plt.savefig("fig1.pdf")
plt.show()
'''
'''

for i in range(n_max):
	spin_conf = gen_spin_conf(i+2)
	print(i+2)
	for j in range(beta_vals.size):
		eigen_val_vec[i][j] = np.log(max_eigen_value_fast(beta_vals[j], P_fast, spin_conf, B_ex))
		eigen_val_vec_b[i][j] = np.log(max_eigen_value_fast(beta_const, P_fast, spin_conf, B_vals[j]))
	if i+2 <= 5:
		for k in range(beta_vals.size):
			spes_heat_vec[i] = np.gradient(np.gradient(eigen_val_vec[i], beta_vals), beta_vals)*(beta_vals[k]/(i+2))
	eigen_val_vec_d[i] = np.gradient(eigen_val_vec_b[i], B_vals)/((i+2)*beta_const)
	plt.plot(beta_vals, eigen_val_vec[i])
	print(i+1," done", n_max - i -1, " to go.")
plt.savefig("fig1.pdf")
plt.show()



for elem in eigen_val_vec_d:
	plt.plot(B_vals, elem)
plt.savefig("fig2.pdf")
plt.show()


for elem in spes_heat_vec:
	plt.plot(beta_vals, elem)
plt.savefig("fig3.pdf")
plt.show()



'''
