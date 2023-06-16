'''
This the factorization problem Hamiltonian based on the block division method 

'''

import numpy as np
import sympy as sy
from sympy import *
from re import sub
import re
from scipy import sparse
import numpy as np
import time
np.set_printoptions(threshold=np.inf)
class Factorization:
	def __init__(
		self,
		number,
		p,
		q,
		width

	):
		self.number = number
		self.width = width
		self.p = p
		self.q = q
		self.p_bit = int(np.floor(np.log2(self.p))) # the total bit length is p_bit +1
		self.q_bit = int(np.floor(np.log2(self.q))) #  p < q
		self.number_bit = int(np.floor(np.log2(self.number)))


		self.row_number = self.q_bit+1 # row number depends on q_bit length
		self.column_number = self.q_bit+self.p_bit+1 # column number depends on both p and q
		self.bk_number = int(np.floor((self.column_number-1)/self.width))
		self.Binary = bin(self.number)[2:]

	# block deviation 

	#def divide_block(self):

		bk_sum = np.zeros(self.bk_number+1, dtype=int) # assuming all the variables are 1. i.e. maximum block value
		carry = np.zeros(self.bk_number+1, dtype=int) # the carry in each block carry[i] is carry calculated from block[i-1]
		self.carryvar_num = np.zeros(self.bk_number+2, dtype=int)


		decrease = 1 
		for i in range(1,1+self.bk_number): #block 1,2,3,4 ……
			for j in range(self.width):
				if ((i+1)+(i-1)*(self.width-1)+j <= self.q_bit+1):
					bk_sum[i] +=((i+1)+(i-1)*(self.width-1)+j)* 2**j
				else:
					bk_sum[i] += ((i+1)+(i-1)*(self.width-1)+j-decrease*2)* 2**j
					decrease += 1

			bk_sum[i] += carry[i-1] # add all the multiplication terms and carry term

			MaxCarry = int(np.floor(bk_sum[i]/2**(self.width)))
			if MaxCarry ==0 :
				self.carryvar_num[i] = 0
			else:
				self.carryvar_num[i] = int(np.floor(np.log2(MaxCarry)))+1 # carry variable numbers
			carry[i] = 2**self.carryvar_num[i]-1 # generate the next block carry total value 
		
		self.sum_carrynum = 0

		if  self.number_bit - self.bk_number*self.width >= 2:
			for i in range(1,self.bk_number+1): 
				self.sum_carrynum+=self.carryvar_num[i]
			self.prodfi_num = self.bk_number+1 # the i range in calculating prodf
		else:
			self.carryvar_num[self.bk_number] = 0
			for i in range(1,self.bk_number): # ignore the last block self.carryvar_number 
				self.sum_carrynum+=self.carryvar_num[i]
			self.prodfi_num = self.bk_number # the i range in calculating prodf

		additional_num = (self.p_bit-1)*(self.q_bit-1) # additional bits number to reduce high order bit

		self.totalbit = self.sum_carrynum+additional_num+self.p_bit-1+self.q_bit-1
		#print("the total bit number is "+str(self.sum_carrynum+additional_num+self.p_bit-1+self.q_bit-1))
		self.p_number = self.p_bit-1
		self.q_number = self.q_bit-1
		#return(self.prodfi_num,self.carryvar_num,self.sum_carrynum,additional_num,self.p_bit-1,self.q_bit-1,self.totalbit)
		

	def cost_function(self):
		#prodfi_num,self.carryvar_num,self.sum_carrynum,_,_,_,_ =  Factorization.divide_block(self)
		p = np.array(np.zeros(self.prodfi_num*self.width+1),dtype = object)
		q = np.array(np.zeros(self.prodfi_num*self.width+1),dtype = object)
		c = np.array(np.zeros(self.sum_carrynum+1),dtype = object)
		
		ProdF = np.array(np.zeros(self.prodfi_num+1),dtype = object) #sum of product terms
		CinF = np.array(np.zeros(self.prodfi_num+1),dtype = object) # sum of carry variables of block i 
		FcaF= np.array(np.zeros(self.prodfi_num+1),dtype = object) # sum of carry variables of block i+1
		TarVal = np.array(np.zeros(self.prodfi_num+1),dtype = object)
		cost = np.array(np.zeros(self.prodfi_num+1),dtype = object)
		total_cost = Symbol('')


		carryvar_sum = np.zeros(len(self.carryvar_num),dtype = int)
		for i in range(0,len(self.carryvar_num)):
			for j in range(i+1):
				carryvar_sum[i] += self.carryvar_num[j] # carryvar_sum[i] the total carry variable number before i block (include)


		for i in range(self.prodfi_num*self.width+1): 
			p[i] = ( "p"+str(i))
		for i in range(self.prodfi_num*self.width+1):
			q[i] = ( "q"+str(i))
		for i in range(self.sum_carrynum+1):
			c[i] = ( "c"+str(i))

		for i in range(1,self.prodfi_num+1): # total block number is prodfi_num
			for j in range(1+(i-1)*self.width,1+i*self.width):
				for m in range(j+1):
					if m <= self.p_bit:
						n = j-m 
						if n <= self.q_bit:
							ProdF[i] += 2**(j-1-(i-1)*self.width)*Symbol(p[m])*Symbol(q[n])# Prodf_i
			for ca in range(1,self.carryvar_num[i-1]+1):
				if (i-2) < 0:
					CinF[i] +=  2**(ca-1)*Symbol(c[carryvar_sum[0]+ca])
				else:
					CinF[i] +=  2**(ca-1)*Symbol(c[carryvar_sum[i-2]+ca])#CinF_i
			for fca in range(1,self.carryvar_num[i]+1):
				FcaF[i] += 2**(self.width+fca-1)*Symbol(c[self.carryvar_num[i-1]+fca])

			TarVal[i] = int(self.Binary[-(i)*self.width-1:-(i-1)*self.width-1 ], 2)

		if self.carryvar_num[self.bk_number] == 0:
			ProdF[self.prodfi_num] = 0
			TarVal[self.prodfi_num] = int(self.Binary[-len(self.Binary):-(i-1)*self.width-1 ], 2)
			for j in range(1+(self.prodfi_num-1)*self.width,1+self.column_number):
				for m in range(j+1):
					if m <= self.p_bit:
						n = j-m 
						if n <= self.q_bit:
							ProdF[self.prodfi_num] += 2**(j-1-(self.prodfi_num-1)*self.width)*Symbol(p[m])*Symbol(q[n])# Prodf_i
		#print(ProdF[self.prodfi_num])
		for i in range(1,self.prodfi_num+1):
			cost[i] = (ProdF[i]+CinF[i]-FcaF[i]-TarVal[i])**2 
			cost[i] = cost[i].subs({p[0]:1,p[self.p_bit]:1,q[0]:1,q[self.q_bit]:1})
			#print(cost[i])
			cost[i] = sub(r'\*\*2','',str(expand(cost[i])))
			total_cost +=  Symbol(cost[i])
		#print(simplify(str(total_cost)))
		return(simplify(str(total_cost)))#string

	def transf(self,sign,alpha,t,a,b,c):
		if sign == 1:
			return(alpha*(Symbol(t)*Symbol(c)+2*(Symbol(a)*Symbol(b)-2*Symbol(a)*Symbol(t)-2*Symbol(b)*Symbol(t)+3*Symbol(t))))
		if sign == -1:
			return(alpha*(-Symbol(t)*Symbol(c)+2*(Symbol(a)*Symbol(b)-2*Symbol(a)*Symbol(t)-2*Symbol(b)*Symbol(t)+3*Symbol(t))))

	def auxiliary_replacement(self):
		total_cost = Factorization.cost_function(self)
		t = np.array(np.zeros((self.p_bit-1)*(self.q_bit-1)+1),dtype = object)
		for i in range(1,(self.p_bit-1)*(self.q_bit-1)+1):  # why here 2 times  
			t[i] = "t"+str(i)


		k = 1
		#total_cost = '+ 2*p1*q2*t3 '
		#print(total_cost)
		for m in range(1,self.p_bit):
			for n in range(1,self.q_bit):

				#+ p_ap_b q_a q_b
				pat = r'\+\s(\d+)\*(\w+?)\*'+str('p'+str(m))+r'\*'+str('q'+str(n))+r'\*(\w+?)'+' '
				aaa = re.findall(pat,str(total_cost))
				find_len = len(aaa)

				for i in range(find_len):
					patt = r'\+\s(\d+)\*'+str((aaa[i])[1])+r'\*'+str('p'+str(m))+r'\*'+str('q'+str(n))+r'\*'+str((aaa[i])[2])
					kkk = Factorization.transf(self,1,int((aaa[i])[0]),str('t'+str(k)),str('p'+str(m)),str('q'+str(n)),str((aaa[i])[1])+'*'+str((aaa[i])[2]))
					total_cost = sub(patt,'+ '+str(kkk),str(total_cost))

				# - p_ap_b q_a q_b
				pat = r'\-\s(\d+)\*(\w+?)\*'+str('p'+str(m))+r'\*'+str('q'+str(n))+r'\*(\w+?)'+' '
				aaa = re.findall(pat,str(total_cost))
				find_len = len(aaa)

				for i in range(find_len):
					patt = r'\-\s(\d+)\*'+str((aaa[i])[1])+r'\*'+str('p'+str(m))+r'\*'+str('q'+str(n))+r'\*'+str((aaa[i])[2])
					kkk = Factorization.transf(self,-1,int((aaa[i])[0]),str('t'+str(k)),str('p'+str(m)),str('q'+str(n)),str((aaa[i])[1])+'*'+str((aaa[i])[2]))
					total_cost = sub(patt,str(kkk),str(total_cost))
				k += 1
		k =1 
		for m in range(1,self.p_bit):
			for n in range(1,self.q_bit):
				#+ pm*qn*?
				pat = r'\+\s(\d+)\*'+str('p'+str(m))+r'\*'+str('q'+str(n))+r'\*(\w+?) '
				aaa = re.findall(pat,str(total_cost))
				#print(aaa)

				find_len = len(aaa)
				for i in range(find_len):
					patt = r'\+\s(\d+)\*'+str('p'+str(m))+r'\*'+str('q'+str(n))+r'\*'+str((aaa[i])[1])
					kkk = Factorization.transf(self,1,int((aaa[i])[0]),str('t'+str(k)),str('p'+str(m)),str('q'+str(n)),str((aaa[i])[1]))
					total_cost = sub(patt,'+ '+str(kkk),str(total_cost))
				
				#+ ?*pm*qn
				pat_2 = r'\+\s(\d+)\*(\w+?)\*'+str('p'+str(m))+r'\*'+str('q'+str(n))+' '
				bbb = re.findall(pat_2,str(total_cost))
				#print(bbb)

				find_len = len(bbb)
				for i in range(find_len):
					patt_2 = r'\+\s(\d+)\*'+str((bbb[i])[1])+r'\*'+str('p'+str(m))+r'\*'+str('q'+str(n))+' '
					zzz = Factorization.transf(self,1,int((bbb[i])[0]),str('t'+str(k)),str('p'+str(m)),str('q'+str(n)),str((bbb[i])[1]))
					total_cost = sub(patt_2,'+ '+str(zzz),str(total_cost))
					#print(str(zzz))


				#- pm*qn*?
				pat_3 = r'\-\s(\d+)\*'+str('p'+str(m))+r'\*'+str('q'+str(n))+r'\*(\w+?) '
				ccc = re.findall(pat_3,str(total_cost))
				#print(ccc)
				find_len = len(ccc)
				for i in range(find_len):
					patt = r'\-\s(\d+)\*'+str('p'+str(m))+r'\*'+str('q'+str(n))+r'\*'+str((ccc[i])[1])
					xxx = Factorization.transf(self,-1,int((ccc[i])[0]),str('t'+str(k)),str('p'+str(m)),str('q'+str(n)),str((ccc[i])[1]))
					total_cost = sub(patt,str(xxx),str(total_cost))

				#- ?*pm*qn
				pat_4 = r'\-\s(\d+)\*(\w+?)\*'+str('p'+str(m))+r'\*'+str('q'+str(n))+' '
				ddd = re.findall(pat_4,str(total_cost))
				#print(ddd)
				find_len = len(ddd)
				for i in range(find_len):
					patt = r'\-\s(\d+)\*'+str((ddd[i])[1])+r'\*'+str('p'+str(m))+r'\*'+str('q'+str(n))+' '
					zzz = Factorization.transf(self,-1,int((ddd[i])[0]),str('t'+str(k)),str('p'+str(m)),str('q'+str(n)),str((ddd[i])[1]))
					#print(zzz)
					total_cost = sub(patt,str(zzz),str(total_cost))

				k += 1

		total_cost = sympify(total_cost)
		#print(total_cost)

		return(str(total_cost)) # change object to string

	def generate_matrix(self,word,siteofword,totalsite):
		if siteofword != totalsite:
			word_len = len(word)
			if word_len == 1:
				word.insert(0,'1') # for the case '+ 8*q3*q6 - q3' no pre-factor in q3
		word_len = len(word)
		sigma_Z = np.array([[1, 0],[0, -1]])
		ide = np.eye(2)
		binary = 1/2*(ide-sigma_Z)
		#_,_,self.sum_carrynum,additional_num,self.p_number,self.q_number,self.totalbit = Factorization.divide_block(self)
		A = np.ones(2**self.totalbit, dtype=int)
		#print(word_len)
		for i in range(1,word_len):
			split_word =  re.findall(r'(\w)',word[i])
			charact = split_word[0]
			site = int(split_word[1])


			if charact == 'p':
				P = binary
				for i in range(int(site)-1):
					P = sparse.kron(ide,P).toarray()
				for i in range(self.totalbit-int(site)):
					P = sparse.kron(P,ide).toarray()
				A = A*np.diag(P)

			if charact == 'q':
				Q = binary
				for i in range(self.p_number+int(site)-1):
					Q = sparse.kron(ide,Q).toarray()
				for i in range(self.totalbit-int(site)-self.p_number):
					Q = sparse.kron(Q,ide).toarray()
				A = A*np.diag(Q)

			if charact == 'c':
				C = binary
				for i in range(self.p_number+self.q_number+int(site)-1):
					C = sparse.kron(ide,C).toarray()
				for i in range(self.totalbit-int(site)-self.p_number-self.q_number):
					C = sparse.kron(C,ide).toarray()
				A = A*np.diag(C)



			if charact == 't':
				T = binary
				for i in range(self.p_number+self.q_number+self.sum_carrynum+int(site)-1):
					T = sparse.kron(ide,T).toarray()
				for i in range(self.totalbit-int(site)-self.p_number-self.q_number-self.sum_carrynum):
					T = sparse.kron(T,ide).toarray()
				A = A*np.diag(T)
		#print(int(word[0]))

		return(int(word[0])*A)


	def sign_func(self,sign):
		if sign == '+':
			return(+1)
		if sign == '-':
			return(-1)

	def Hamiltonian_matrix(self):
		#_,_,self.sum_carrynum,additional_num,self.p_number,self.q_number,self.totalbit = Factorization.divide_block(self)
		reduced_cost = Factorization.auxiliary_replacement(self)
		reduced_cost = reduced_cost.replace(' ','')
		#print(reduced_cost)
		a = re.split(r'([\-|\+])', reduced_cost)
		#print(a)
		len_a  = len(a)
		b = [['+']]
		for i in range(len_a):
			b.append(re.split(r'\*',a[i]))
		#print(b)
		Hamiltonian = np.zeros(2**self.totalbit, dtype=float)
		j = 0

		#starttime = time.clock()

		for i in range(int(len(b)/2)): # minus 1 to ignore the last constant
			Hamiltonian += Factorization.sign_func(self,b[j*2][0])*Factorization.generate_matrix(self,b[j*2+1],j*2+1,int(len(b)-1))
			j += 1
			endtime = time.clock()

		#usetime = (endtime - starttime)
		#norm = 1
		#norm = np.linalg.norm(Hamiltonian,ord = np.inf)
		#print(Hamiltonian/norm)
		#print(usetime)
		return(Hamiltonian)



if __name__ == '__main__':
	tc = Factorization(55,5,11,3)
	#tc = Factorization(49,7,7,3)
	#tc = Factorization(35,5,7,2)
	#tc = Factorization(143,11,13,2)
	#tc = Factorization(5*61,5,61,3)
	#tc = Factorization(1073,29,37,3)
	#tc = Factorization(187,11,17,3)
	#tc = Factorization(17*13,13,17,3)
	#tc = Factorization(2449,31,79,3)
	#tc = Factorization(59989,251,239,3)
	#tc = Factorization(1517,37,43,3)
	#tc = Factorization(1005973,997,1009,3)
	#tc.divide_block()
	#tc.cost_function()
	#tc.auxiliary_replacement()
	#tc.generate_matrix()
	tc.Hamiltonian_matrix()