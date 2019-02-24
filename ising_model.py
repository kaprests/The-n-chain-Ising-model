import numpy as np
from matplotlib import pyplot as plt
import time

start = time.time()

B_ex = 2
J_par = 1
J_tan = 1
n_max = 8 
beta_const = 0.5
kb = 1.3806485279e-23


def gen_spin_conf(n):
	spin_conf = np.ones([2**n, n])
	for i in range(n):
		spin_conf.T[i] = np.array([[1]*2**(i) + [-1]*2**(i)] * ((2**n)//(2**(i+1)))).flatten()
	return spin_conf


def gen_P(spin_conf, Be, beta, Jp, Jt):
	dim = spin_conf.shape[0]
	A = beta*Jp * np.matmul(spin_conf, spin_conf.T)
	B = beta*Jt * np.repeat(np.sum(spin_conf * np.apply_along_axis(np.roll, 1, spin_conf, 1), axis=1).reshape(dim,1), dim, axis=1)
	sum_vec = np.sum(spin_conf, axis=1)
	C = (beta*Be/2) * (np.tile(sum_vec, [dim, 1]).T + sum_vec.T)
	return np.exp(A + B + C)


def max_eigen_value(beta, gen_P, spin_conf, Be, Jp=J_par, Jt=J_tan):
	e_vals = np.linalg.eigvals(gen_P(spin_conf, Be, beta, Jp, Jt))
	return np.amax(np.real(e_vals))


beta_vals = np.linspace(0.005, 1, 100)
B_vals = np.linspace(-1, 1, 100)
T_vals = 1/(kb*beta_vals)
eigen_val_vec = np.zeros([n_max, 100])
eigen_val_vec_b = np.zeros([n_max, 100])
eigen_val_vec_d = np.zeros([n_max, 100])
spes_heat_vec = np.zeros([4, 100])

for i in range(n_max):
	spin_conf = gen_spin_conf(i+1)
	print("n: ",i+1)
	for j in range(beta_vals.size):
		eigen_val_vec[i][j] = np.log(max_eigen_value(beta_vals[j], gen_P, spin_conf, B_ex))
		eigen_val_vec_b[i][j] = np.log(max_eigen_value(beta_const, gen_P, spin_conf, B_vals[j]))
	if i+1 <= 5 and i+1 >= 2:
		for k in range(beta_vals.size):
			spes_heat_vec[i-1] = np.gradient(np.gradient(eigen_val_vec[i], beta_vals), beta_vals)*((beta_vals[k]**2)/(i+2))
	eigen_val_vec_d[i] = np.gradient(eigen_val_vec_b[i], B_vals)/((i+2)*beta_const)
	print(i+1," done", n_max - i -1, " to go.")
	plt.plot(beta_vals, eigen_val_vec[i], label="n = " + str(i+1))
plt.title("Largest eigenvalues -  B=" + str(B_ex) + ", Jp=" + str(J_par) + ", Jt=" + str(J_tan))
plt.legend()
plt.xlabel("beta - 1/kb*T")
plt.ylabel("Eigenvalue")
plt.savefig("fig1.pdf")

end = time.time()
print(end - start)

plt.show()

for i in range(eigen_val_vec_d.shape[0]):
	plt.plot(B_vals, eigen_val_vec_d[i], label="n=" + str(i+1))
plt.title("Magnetization per spin - beta = " + str(beta_const)  + ", Jp=" + str(J_par) + ", Jt=" + str(J_tan))
plt.legend()
plt.xlabel("External magnetic field - B")
plt.ylabel("Magnetizatio per spin - m")
plt.savefig("fig2.pdf")
plt.show()

for i in range(spes_heat_vec.shape[0]):
	# Not sure what's best of plotting with regards to T or beta(T)
	plt.plot(T_vals, spes_heat_vec[i], label="n=" + str(i+2))
plt.title("Specific heat per spin -  B=" + str(B_ex) + ", Jp=" + str(J_par) + ", Jt=" + str(J_tan))
plt.legend()
plt.xlabel("Temperature - T")
plt.ylabel("Spesific heat per spin - C_B")
plt.savefig("fig3.pdf")
plt.show()
