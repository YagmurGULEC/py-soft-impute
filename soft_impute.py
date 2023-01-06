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
import pandas as pd
import matplotlib.pyplot as plt

def generate_data(nrows,ncols,percentage):
    ncol=int(ncols)
    nrow=int(nrows)
    data=np.random.randn(nrow,ncol)
    data_scaled= ((data - data.min()) * (1/(data.max() - data.min()) * 255))
    
    number_nans=int(ncol*nrow*int(percentage)/100)
    np.put(data_scaled,np.random.choice(data.size,number_nans,replace=False),np.nan)
    return data_scaled
    
#Base class and ISTA
class Impute:
    def __init__(self,thresh=1e-10,beta=0.01, maxit=400, random_state=None, verbose=False):
        #observed values
        self.omega=None
        self.Y=None
        self.X=None
        self.beta=beta
        self.maxit = maxit
        self.rs = np.random.RandomState(random_state)
        self.verbose = verbose
        self.thresh=thresh
        #convergence rate
        self.epsilon=[]
        #cost function
        self.cost=[]
   
    def SVST(self,X,beta):
        U,s,V=np.linalg.svd(X)
        sthresh=np.maximum(s-beta,0)
        B = U@la.diagsvd(sthresh,*X.shape)@V
        return B  
         
    def cost_fun(self):
        #nuclear norm
        nucnorm=la.norm(self.X,'nuc')
        #frobenius norm+nuclear norm
        costfun=0.5*la.norm(self.X[self.omega]-self.Y[self.omega])**2+self.beta*nucnorm
        return costfun

    def get_residual(self):
        df=pd.DataFrame(columns=['cost','convergence'])
        df['cost']=self.cost
        df['convergence']=self.epsilon
        return df

    def calculate_convergence_rate(self,Xnew,Xold):
        return (la.norm(Xnew-Xold)**2/la.norm(Xold)**2)

    def fit(self, X):
        self.omega=np.where(~np.isnan(X))
        self.Y=X.copy()
        self.Y[np.isnan(X)]=0
        iters = 0
        Xold=self.Y
        ratio=1
        while iters < self.maxit and ratio>self.thresh:
            iters += 1
            Xnew=self.SVST(Xold,self.beta)
            ratio=self.calculate_convergence_rate(Xnew,Xold)
            Xold=Xnew
            self.X=Xold
            self.cost.append(self.cost_fun())
            self.epsilon.append(ratio)
        if self.verbose:
            residual=self.get_residual()
            print (residual)
        return self
    
    #return imputed matrix
    def transform(self):
        return self.X

"""
Fast Iterative Soft-Thresholding Algorithm (FISTA)
Modification of ISTA to include Nesterov acceleration for faster convergence.
"""
class FISTA(Impute):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
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
        ratio=1
        while iters < self.maxit and ratio>self.thresh:
            iters += 1
            self.Z[self.omega]=self.Y[self.omega]
            Xnew=self.SVST(self.Z,self.beta)
            t = (1 + np.sqrt(1+4*told**2))/2
            self.Z=Xnew+((told-1)/t)*(Xnew-Xold)
            ratio=self.calculate_convergence_rate(Xnew,Xold)
            Xold=Xnew
            told=t
            self.X=Xold
            self.cost.append(self.cost_fun())
            self.epsilon.append(ratio)
        return self  
      
#Alternating directions method of multipliers (ADMM) algorithm
class ADMM(Impute):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
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
        ratio=1
        while iters < self.maxit and ratio>self.thresh:
            iters += 1
            self.Z=self.SVST(Xold+self.L,self.beta/self.mu)
            Xnew=(self.Y+self.mu*(self.Z-self.L))/(self.mu+omega_matrix)
            self.L = self.L + Xnew - self.Z
            ratio=self.calculate_convergence_rate(Xnew,Xold)
            Xold=Xnew
            self.X=Xold
            self.cost.append(self.cost_fun())
            self.epsilon.append(ratio)
        return self

def main():
    clf=FISTA(beta=0.01,maxit=50,verbose=True)
    #X=generate_data(100,10,10)
    X=np.array([[1,np.nan,3],[4,5,6],[7,8,9],[10,11,12]])
    clf.fit(X)
    Ximputed=clf.transform()
    print (Ximputed)
if __name__ == '__main__':
    main()

