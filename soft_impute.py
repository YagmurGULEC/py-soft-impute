'''
   Copyright 2015 Travis Brady

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
'''
from __future__ import print_function
import numpy as np
import scipy.linalg as la
from scipy.linalg import svdvals,norm

class Impute:
    def __init__(self,thresh=1e-05,beta=0.01, maxit=400, random_state=None, verbose=False):
        self.omega=None
        self.Y=None
        self.X=None
        self.beta=beta
        self.maxit = maxit
        self.rs = np.random.RandomState(random_state)
        self.verbose = verbose
        self.cost=[]
       
        
    def SVST(self,X,beta):
        U,s,V=np.linalg.svd(X)
        sthresh=np.maximum(s-beta,0)
        B = U@la.diagsvd(sthresh,*X.shape)@V
        return B  
         
    def cost_fun(self):
        nucnorm=np.sum(svdvals(self.X))
        costfun=0.5*norm(self.X[self.omega]-self.Y[self.omega])**2+self.beta*nucnorm
        return costfun
    
    def fit(self, X):
        self.omega=np.where(~np.isnan(X))
        self.Y=X.copy()
        self.Y[np.isnan(X)]=0
        self.X=self.Y.copy()
     
        iters = 0
        while iters < self.maxit:
            iters += 1
            self.X[self.omega]=self.Y[self.omega]
            self.X=self.SVST(self.X,self.beta)
            self.cost.append(self.cost_fun())
        return self

    def predict(self, X, copyto=False):
        return self.X,self.cost
"""
Fast Iterative Soft-Thresholding Algorithm (FISTA)
Modification of ISTA to include Nesterov acceleration for faster convergence.
"""
class FISTA(Impute):
    def __init__(self):
        super().__init__(maxit=200)
        self.Z=None
    
    def fit(self,X):
        self.omega=np.where(~np.isnan(X))
        self.Y=X.copy()
        self.Y[np.isnan(X)]=0
        self.X=self.Y.copy()
        self.Z=self.X.copy()
        told=1
        Xold=self.Y.copy()
        iters = 0
        while iters < self.maxit:
            iters += 1
            self.Z[self.omega]=self.Y[self.omega]
            self.X=self.SVST(self.Z,self.beta/self.mu)
            t = (1 + np.sqrt(1+4*told**2))/2
            self.Z=self.X+((told-1)/t)*(self.X-Xold)
            Xold=self.X
            told=t
            self.cost.append(self.cost_fun())
        return self    
#Alternating directions method of multipliers (ADMM) algorithm
class ADMM(Impute):
    def __init__(self):
        super().__init__(maxit=50)
        self.Z=None
        self.L=None
        self.mu=self.beta
    def fit(self,X):
        self.omega=np.where(~np.isnan(X))
        self.Y=X.copy()
        self.Y[np.isnan(X)]=0
        self.X=self.Y.copy()
        self.Z=np.zeros(X.shape)
        self.L=np.zeros(X.shape)
        omega_matrix=np.zeros(X.shape)
        omega_matrix[self.omega]=1
        told=1
        Xold=self.Y.copy()
        iters = 0
        while iters < self.maxit:
            iters += 1
            self.Z=self.SVST(self.X+self.L,self.beta/self.mu)
            self.X=(self.Y+self.mu*(self.Z-self.L))/(self.mu+omega_matrix)
            self.L = self.L + self.X - self.Z
        return self   
def main():
    clf=ADMM()
    X=np.array([[1,np.nan, 3], [4,5,6],[7, 8,9],[10, 11, 12]])
   
    clf.fit(X)
    X,cost=clf.predict(X)
    print (clf.maxit)

if __name__ == '__main__':
    main()

