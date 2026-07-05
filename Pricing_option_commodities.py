# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 00:21:26 2026

@author: sbanerj1
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

## implementing binomial-tree model to price both European and American option 
def BionomialEuroAmerica(N, K, T, sigma, S0, r, delta, American):
    
    dt=T/N
    
    u=np.exp(sigma*np.sqrt(dt))
    d=np.exp(-sigma*np.sqrt(dt))
    
    p=(np.exp((r-delta)*dt) - d)/(u-d)
    
    S=np.zeros(N+1)
     
    for j in range(N+1):
        S[j]=S0*u**j*d**(N-j)
        
    V=np.maximum(K-S,0)
    Regime=[]
   # Boundary=np.zeros(N)
    Boundary=np.full(N, np.nan)
    
    
    for i in range(N-1,-1, -1):
        
        if (American):
            
            Si=np.zeros(i+1)
            exercise_nodes=[]
            
        for j in range(i+1):
        
             V[j]= np.exp(-(r)*dt)*(p*V[j+1]+(1-p)*V[j])
        
             if (American):
                 
                Si[j]=S0*u**j*d**(i-j)
                Exercise=np.maximum(K-Si[j],0)
              
                
                if Exercise>V[j]:
                    Regime.append((i,j))
                    exercise_nodes.append(Si[j])
              
                V[j]=np.maximum(V[j], Exercise)   # Early exercise feature of American option is implemented
        
        if American and exercise_nodes:  
            Boundary[i]=max(exercise_nodes)
                
    return (V[0], Regime, Boundary)

# implementing the Black-Scholes formula
def BlackScholesFormula(K, T, sigma, S0, r, delta):
    
    d1=(np.log(S0/K)+(r-delta+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    
    V=K*np.exp(-r*(T))*st.norm.cdf(-d2)-np.exp(-delta*T)*S0*st.norm.cdf(-d1)
    
    return(V)
    
def mainCalculation():
    
    # Parameters
    T=5
    S0=1 
    K=1.6
    sigma=0.4
    r=0.12
    delta=0.05
    
    N=[5, 10, 50, 100, 300, 600]
    Price=np.zeros(len(N))
    
    
    # Compare pricing of European Option using Binomial Tree model
    # and Black Scholes equation
    for i in range(len(N)):
    
        Price[i],_,_ =BionomialEuroAmerica(N[i], K, T, sigma, S0, r, delta, American=False)
        
    PriceBS=BlackScholesFormula(K, T, sigma, S0, r, delta)
    
    plt.figure()
    plt.plot(N,Price, 'o-', label='Binomial')
    plt.axhline(y=PriceBS, color='r', linestyle='--', label='Black-Scholes')
    plt.xlabel('N')
    plt.ylabel('European Option Price')
    plt.legend()

    
    ## Plotting the impact of changing delta on European and American option
    delta=[0.02, 0.03, 0.05, 0.07]
    N=600
    PriceA=np.zeros(len(delta))
    PriceE=np.zeros(len(delta))
    
    plt.figure()
    for i in range(len(delta)):

        PriceE[i],_,_ = BionomialEuroAmerica(N, K, T, sigma, S0, r, delta[i], American=False)
        PriceA[i], regime, boundary =BionomialEuroAmerica(N, K, T, sigma, S0, r, delta[i], American=True)
        plt.plot(range(0,N), boundary, linewidth=0.8, label="$\delta$="+ str(delta[i]))
    
    # plotting how the early exercise boundary changes with changing delta
    plt.xlabel("Time step i")
    plt.ylabel("Early exercise boundary")
    plt.title("Early exercise boundary with changing $\delta$")
    plt.legend()
    
     # all the i and j values where early exercise becomes optimal    
    i_vals=[x[0] for x in regime]
    j_vals=[x[1] for x in regime]
    
    # plotting the regime of early exercise for the largest \delta value         
    plt.figure()
    plt.scatter(i_vals,j_vals)
    plt.xlabel("Time step i")
    plt.ylabel("Node index j")
    plt.title("Early exercise region (δ = "+ str(delta[i])+")")
     
    # plotting how changing delta changes the European and American option price
    plt.figure()
    plt.plot(delta, PriceA, 'o-', label='Binomial-American')
    plt.plot(delta, PriceE, 'o-', label='Binomial-European')
    plt.xlabel('delta')
    plt.ylabel('Option Price')
    plt.legend()

    
mainCalculation()
