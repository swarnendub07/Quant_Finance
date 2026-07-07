# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 10:45:08 2026

@author: sbanerj1
"""

import numpy as np
import matplotlib.pyplot as plt


def COS_density(x, N, a, b, S0, gamma, T):
# Recovering probability density from characteristic function
# using COS method
    
    
    k=np.arange(N)  # assign values to k= 0,1,2...,N-1 
    
    u=k*(np.pi)/(b-a) # u=(k*pi)/(b-a)
    
    # characteristic function
    phi=np.exp(1j*u*S0/(1-0.5*1j*u*gamma**2*T)) 
    
    # coefficients of the cos series which gives the density 
    # at a particular point x
    Fk=2/(b-a)*np.real(phi*np.exp(-1j*a*u))
    Fk[0]=Fk[0]*0.5
    
    density=np.zeros(len(x))
    
    # calculating density as a function of x, for a paricular T
    for i in range(0,len(x)):

        density[i]=np.sum(Fk*np.cos(k*np.pi*(x[i]-a)/(b-a)))

    return density

## 
def MonteCarloPaths(gamma, T, S0, NoOfPaths, NoOfSteps):
    
    Z=np.random.normal(0,1,[NoOfPaths, NoOfSteps])
    S=np.zeros([NoOfPaths, NoOfSteps+1])
    time=np.zeros([NoOfSteps+1]) 
    
    S[:,0]=S0
    
    dt=T/NoOfSteps
    
    for i in range(0, NoOfSteps):
        S[:,i+1]=S[:,i]+ gamma* np.sqrt(S[:,i])*np.sqrt(dt)*Z[:,i]
        
        time[i+1]=time[i]+dt

    return S[:,-1]
    
   
def mainCalculation():
    
    S0=1 
    T=5
    gamma=0.25
    
    a=0 
    b=4
    
    N=50
    x=np.linspace(a, b, 100)
    
    NoOfPaths=10000
    NoOfSteps=100
    
    density=COS_density(x, N, a, b, S0, gamma, T)
    
    MCdensity= MonteCarloPaths(gamma, T, S0, NoOfPaths, NoOfSteps)
    
    plt.figure()
    plt.plot(x,density)
    plt.xlabel("S")
    plt.ylabel("Density") 
    
   
    plt.hist(MCdensity, bins=100, density=True, alpha=0.6)
    plt.xlabel("S")
    plt.ylabel("Density")
    
    plt.show()

mainCalculation()

        
        
        
        
        