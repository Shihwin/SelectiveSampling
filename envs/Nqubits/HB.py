'''
This is factorization Beginning Hamiltonian HB = ∑ _i[1 − X_i]/2

'''

#import cupy as np
#from cupyx.scipy.sparse import csr_matrix
import numpy as np
from scipy.sparse import csr_matrix
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def HB(n):
    sigma_x = np.array([[0,1],[1,0]])
    ones = np.eye(2,2)
    B = 0
    for i in range (1,n+1):
        A = 1/2*(ones-sigma_x)
        for j in range(2,i+1):
            A= np.kron(ones,A)
        for k in range(1,n-i+1):
            A= np.kron(A,ones)
        B=B+A
    #B = 30*csr_matrix(B)
    return(csr_matrix(B))

if __name__ == '__main__':
    print(__name__)
    print(HB(2))
