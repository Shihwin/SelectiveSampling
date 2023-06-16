import multiprocessing as mp
import HB 
import HP
import math
# import cupy as np  
# from cupyx.scipy.sparse import csr_matrix
import numpy as np
from scipy import sparse
import time
import matplotlib.pyplot as plt
import random
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#plt.switch_backend('agg')
def MakeMatrix(n,Numbers):
	lenthNumbers = len(Numbers)
	Hp_array = np.zeros((lenthNumbers,2**n),dtype = float)
	Hb=HB.HB(n)
	for i in range(len(Numbers)):
		number = Numbers[i]
		fact = HP.Factorization(number[0],number[1],number[2],number[3])
		Hp_array[i][:]=fact.Hamiltonian_matrix()
		print('nihao')
	return Hb,Hp_array

def CalcuFidelity(n,p,Hb,Hp_array,T,g):
	lenthNumbers = Hp_array.shape[0]
	T_evolve = T
	g_evolve = g
	Hb_evolve = Hb

	averageSuccess = 0

	for i in range(lenthNumbers):
		Pi = np.pi
		t = 0   
		Hp=Hp_array[i][:]
		norm = np.linalg.norm(Hp,ord = np.inf)
		Hb=Hb_evolve/norm
		Hp = Hp/norm
		g = g_evolve*norm
		T = T_evolve*norm
		C = []
		find_site = []
		N=pow(2,n)
		#print(n)
		ident = np.eye(N)

		for i in range(N):
			if not(Hp[i]).any():
				find_site.append(i)

		lenthsite = len(find_site)

		psi = 1/(2**(n/2))*np.ones(N,dtype=complex)
		for i in range(int(T/g)+1): # the steps of iteration
				c = 0
				t = g*i+g/2
				fourier_expan = np.array([np.sin(1*Pi*t/T),np.sin(2*Pi*t/T),np.sin(3*Pi*t/T),np.sin(4*Pi*t/T),np.sin(5*Pi*t/T),np.sin(6*Pi*t/T)])
				s = (1/T*t)**2+np.dot(p,fourier_expan)
				first = np.exp(-1j*s/2*g*Hp)*psi
			
				second =  first-1j*(1-s)*g*Hb.dot(first)+1/2*(-1j*g*(1-s))**2*Hb.dot(Hb.dot(first))
			
				psi = np.exp(-1j*s/2*g*Hp)*second
				psi = psi/np.linalg.norm(psi)

				for site in range(lenthsite):
					c += float(np.log10((abs(psi[find_site[site]]**2))))

				C.append(c)
		averageSuccess += C[-1]/lenthNumbers
	return averageSuccess

if __name__ == '__main__':
	starttime = time.time()
	averagefidelity = []
	hardinstance = []
	Numbers = [[111, 3, 37, 3],[141, 3, 47, 3],[159, 3, 53, 3],[177, 3, 59, 3]]
	n  = 6
	Hb,Hp_array = MakeMatrix(n, Numbers)
	T = 1
	g = 10**(-3) # interval
	p = np.array([-0.04264595 , 0.08286855 ,-0.02211497 ,-0.01031495,  0.01430113 , 0.0119922])

	C =CalcuFidelity(n,p,Hb,Hp_array,T,g) 
	print(C)


