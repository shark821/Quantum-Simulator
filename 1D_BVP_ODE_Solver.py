# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 06:36:31 2026

@author: 周俊宇
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def _1D_BVP_ODE_solver(BC,type_of_BC,func,start,stop,N,order):
    #func給出各階微分值在方程中的係數
    x_lattice=np.linspace(start, stop, N+1)
    dx=abs(stop-start)/N
    _1_diff_matrix=sp.diags(diagonals=[np.ones(N)/(dx*2)*(-1),np.zeros(N+1),np.ones(N)/(dx*2)],offsets=[-1,0,1],shape=(N+1,N+1),format='csr')
    _2_diff_matrix=sp.diags(diagonals=[np.ones(N)/(dx**2),np.ones(N+1)*(-2)/(dx**2),np.ones(N)/(dx**2)],offsets=[-1,0,1],shape=(N+1,N+1),format='csr')
    diff_matrix_set=[]
    
    coef=func(x_lattice)
    
    for i in range(order+1):
        A=sp.eye(N+1, format='csr')
        if i%2==0:
            for j in range(int(i/2)):
                A=_2_diff_matrix@A
        else:
            for j in range(int((i-1)/2)):
                A=_2_diff_matrix@A
            A=_1_diff_matrix@A
        diff_matrix_set.append(A)
        
    if type_of_BC=="Dirichlet":
        A=sp.csr_matrix((N+1,N+1))
        n=0
        
        for i in diff_matrix_set:   #各階微分矩陣融合
            A+=sp.diags(diagonals=[coef[n]],offsets=[0],shape=(N+1,N+1),format='csr')@i
            n+=1

        
        b=coef[-1].copy() 
        
        A=A.tolil()        #換成Lil加速替換
        
        for i in BC:        #加入邊界條件
           n=int(round((i[0]-start)/dx))
           A[n]=np.zeros(N+1)
           A[n,n]=1
           b[n]=i[1]
        
        A=A.tocsr()       #換回csr
        
        solution=spla.spsolve(A,b)     #求解
        
        return x_lattice,solution

if __name__=="__main__":
    
    def gravity(x_lattice):
        N=len(x_lattice)-1
        c0=np.zeros(N+1)
        c1=np.zeros(N+1)
        c2=np.ones(N+1)
        b=np.ones(N+1)*(-9.8)
        return [c0,c1,c2,b]
    
    start=0
    stop=1
    BC=[[start,10],[stop,0],]
    type_of_BC="Dirichlet"
    func=gravity    
    N=1000
    order=2
    
    x_lattice,solution=_1D_BVP_ODE_solver(BC,type_of_BC,func,start,stop,N,order)
    
    plt.plot(x_lattice,solution)
    plt.show()
