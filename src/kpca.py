import numpy as np
import warnings


class KPCA:
    def __init__(self, X, kernel, d):
        self.X = X
        self.kernel = kernel 
        self.d = d
    
    def _is_pos_semidef(self, x):
        return np.all(x >= 0)

    def __kernel_matrix(self):
        K = []
        r, c = self.X.shape
        for fil in range(c):
            k_aux = []
            for col in range(c):
                k_aux.append(self.kernel(self.X[:, fil], self.X[:, col]))
            K.append(k_aux)
        K = np.array(K)
        
        # Centering K
        ones = np.ones(K.shape)/c
        K = K - ones@K - K@ones + ones@K@ones
        return K
    
    def descomp(self):
        self.K = self.__kernel_matrix()
        eigval, eigvec = np.linalg.eig(self.K)
        if not self._is_pos_semidef(eigval):
            warnings.warn("matrix K is not positive semidefinite")
            
        # Normalize eigenvectors and compute singular values of K
        tuplas_eig = [(np.sqrt(eigval[i]), eigvec[:,i]/np.sqrt(eigval[i]) ) for i in range(len(eigval))]
        tuplas_eig.sort(key=lambda x: x[0], reverse=True)
        return tuplas_eig
    
    def project(self):
        self.tuplas_eig = self.descomp()
        tuplas_eig_dim = self.tuplas_eig[:self.d]
        self.sigma = np.diag([i[0] for i in tuplas_eig_dim])
        self.v = np.array([list(j[1]) for j in tuplas_eig_dim]).T
        self.sigma = np.real_if_close(self.sigma, tol=1)
        self.v = np.real_if_close(self.v, tol=1)
        self.scores = self.sigma @ self.v.T
        return self.scores