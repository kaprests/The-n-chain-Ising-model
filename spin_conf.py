import numpy as np


def gen_spin_conf(n):
    spin_conf = np.ones([2**n, n]) 
    for i in range(n):
        spin_conf.T[i] = np.array([[1]*2**(i) + [-1]*2**(i)] * ((2**n)//(2**(i+1)))).flatten()
    return spin_conf

print(gen_spin_conf(5))
