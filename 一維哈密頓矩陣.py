# -*- coding: utf-8 -*-
"""
Created on Sun Jan  4 06:12:20 2026

@author: 周俊宇
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


#使用Scipy稀疏矩陣功能構建哈密頓算符
def  _1D_Hamiltonian_Operator(x,V_x,m):
    h_bar=1.05457182*(10**(-34))
    N=len(x)
    dx=x[1]-x[0]
    T=sp.diags(diagonals=[np.ones(N-1)/(dx**2),np.ones(N)/(dx**2)*(-2),np.ones(N-1)/(dx**2)],offsets=[-1,0,1],shape=(N,N),format='csr')*(-h_bar**2)/2/m
    V=sp.diags(diagonals=[V_x],offsets=[0],shape=(N,N),format='csr')
    H=T+V
    return H

def _1D_stationary_basis(H,basis_number):
    eigen_energy,basis_function=spla.eigsh(H,k=basis_number,which='SA')
    return eigen_energy,basis_function

#時間相位因子
def _1D_time_phase_factor(eigen_energy):
    h_bar=1.05457182*(10**(-34))
    phase=-1j*eigen_energy/h_bar
    return phase

#在時間t時的各基底相位
def _1D_wavefunction_time_envolve(x,V_x,m,basis_number,t_lattice):
    H=_1D_Hamiltonian_Operator(x,V_x,m)
    
    eigen_energy,basis_function=_1D_stationary_basis(H,basis_number)
    
    time_envolve=np.zeros((len(t_lattice),basis_number,len(x)),dtype=complex)
    n=0
    for t in t_lattice:
        time_envolve[n]=(basis_function*np.exp(_1D_time_phase_factor(eigen_energy)*t)).T
        n+=1
    return time_envolve , eigen_energy , basis_function

if __name__=="__main__":
    x_lattice=np.linspace(-2*10**(-9),2*10**(-9),2001)    
    V=-np.exp(-(x_lattice/10**(-9))**2)*10**(-18) +1*10**(-18)      
    m=9.1093837*10**(-31)#電子質量    
    t_lattice=np.linspace(0,1*10**(-9),2)#時間採樣格點    
    basis_number=5#只取前五個本徵態        
    #同樣的基底不同時間下的波函數圖形(實部)
    time_dependent_basis , eigen_energy , stationary_basis=_1D_wavefunction_time_envolve(x_lattice,V,m,basis_number,t_lattice)#t=10**(-20)
    plt.title("wavefunction")
    for t in range(len(t_lattice)):
        for i in range(basis_number):
            plt.plot(x_lattice,time_dependent_basis[t,i])
        plt.plot(x_lattice,V*(10**16))
        plt.show()
    
    
    print(eigen_energy/1.602*10**(19),"eV")