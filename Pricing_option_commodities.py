# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 00:21:26 2026

@author: sbanerj1
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

## implementing binomial-tree model to price both European and American option 
def BionomialEuroAmerica(N, K, T, sigma, S0, r, delta, A, Ts, American):
    
    dt=T/N
    time=np.linspace(0, T, N)
    delta_t=np.ones(N) * delta+A*np.sin(2*np.pi*time/Ts)
    
    u=np.exp(sigma*np.sqrt(dt))
    d=np.exp(-sigma*np.sqrt(dt)) 
    S=np.zeros(N+1)
     
    for j in range(N+1):
        S[j]=S0*u**j*d**(N-j)
        
    V=np.maximum(K-S,0)
    PayOff=np.maximum(K-S,0)
    
    # created to store the values for each node of the binomial tree for verification
    S_tree=[]
    V_tree=[]
    PayOff_tree=[]
    ExerciseNodes_tree=[]
    
    # binomial tree is initialized
    S_tree.append(np.flip(S).copy())
    V_tree.append(np.flip(V).copy())
    PayOff_tree.append(np.flip(PayOff).copy())
    
    # initializing the boundary and regime
    Boundary=np.full(N, np.nan)
    Regime=[]
    
    for i in range(N-1,-1, -1):
        
        # calculating the probability at each time step
        # especially useful to calculate separately 
        p=(np.exp((r-delta_t[i])*dt) - d)/(u-d)
         
        
        Si=np.zeros(i+1)
        exercise_nodes = np.empty(i+1)
        exercise_nodes[:] = np.nan
            
        for j in range(i+1):
        
             V[j]= np.exp(-(r)*dt)*(p*V[j+1]+(1-p)*V[j])
        
             if (American):
                 
                Si[j]=S0*u**j*d**(i-j)
                Exercise=np.maximum(K-Si[j],0)
                PayOff[j]=Exercise
        
                
                if Exercise>V[j]:
                    Regime.append((i,j))
                    exercise_nodes[j]=Si[j]
                    
                V[j]=np.maximum(V[j], Exercise)   # Early exercise feature of American option is implemented
        
        # store a copy of S and V in the binomial model
        S_tree.append(np.flip(Si).copy())
        V_tree.append(np.flip(V[:i+1]).copy())
        PayOff_tree.append(np.flip(PayOff[:i+1]).copy()) 
        ExerciseNodes_tree.append(np.flip(exercise_nodes[:i+1]).copy())
        
        if American and np.any(~np.isnan(exercise_nodes[:i+1])):
            Boundary[i]=np.nanmax(exercise_nodes[:i+1])
     
        ## to check the values
        # if i==5:
        #    print("delta =", delta)
        #    print("Si =", Si)
        #    print("continuation =",  V[:i+1])
        
        #    print("Exercise =", Boundary[i])
    
                
    return (V[0], Regime, Boundary, S_tree, V_tree, PayOff_tree, ExerciseNodes_tree)

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
    A=0.00    # non-zero values introduces seasonality in delta
    Ts=1
    
    N=[5, 10, 50, 100, 300, 600]
    Price=np.zeros(len(N))
    
    
    # Compare pricing of European Option using Binomial Tree model
    # and Black Scholes equation
    for i in range(len(N)):
    
        Price[i],_,_,_,_,_,_ =BionomialEuroAmerica(N[i], K, T, sigma, S0, r, delta,  A, Ts, American=False)
        
    PriceBS=BlackScholesFormula(K, T, sigma, S0, r, delta)
    
    plt.figure()
    plt.plot(N,Price, 'o-', label='Binomial')
    plt.axhline(y=PriceBS, color='r', linestyle='--', label='Black-Scholes')
    plt.xlabel('N')
    plt.ylabel('European Option Price')
    plt.legend()

    
    ## Plotting the impact of changing delta on European and American option
    delta=[0.02, 0.07]
    N=600
    PriceA=np.zeros(len(delta))
    PriceE=np.zeros(len(delta))
    
    plt.figure()
    for i in range(len(delta)):

        PriceE[i],_,_,_,_,_,_ = BionomialEuroAmerica(N, K, T, sigma, S0, r, delta[i], A, Ts, American=False)
        PriceA[i], regime, boundary, S_tree, V_tree, PayOff_tree, ExerciseNodes_tree =BionomialEuroAmerica(N, K, T, sigma, S0, r, delta[i], A, Ts, American=True)
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
    
    
    # plotting how seasonal delta changes the early exercise boundary compared to fixed delta
    
    plt.figure()
    plt.plot(range(0,N), boundary, linewidth=0.8, label="fixed $\delta$")  
    
    A=0.05
    delta=0.07
    PriceASeason,_,boundary,_,_,_,_ =BionomialEuroAmerica(N, K, T, sigma, S0, r, delta, A, Ts, American=True)
    
    plt.plot(range(0,N), boundary, linewidth=0.8, label="seasonal $\delta$")
    plt.xlabel("Time step i")
    plt.ylabel("Early exercise boundary")
    plt.title("Early exercise boundary with changing $\delta$")
    plt.legend()
    
    # these values are returned only to check values at all the binomial tree nodes
    return(S_tree, V_tree, PayOff_tree, boundary, ExerciseNodes_tree)
    
S_tree, V_tree, PayOff_tree, boundary, ExerciseNodes_tree =mainCalculation()
