# -*- coding: utf-8 -*-
"""
Created on Sat May 30 09:12:38 2026

Exercise 2 Question 1
@author: sbanerj1
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


def Market_PutOptionValueAnalytic(K, r, T, S, t ):
    ## put option value calculated analytically from the GBM with time-dependent sigma

    v=0.36*(T-t)-0.04/3*(np.exp(-3*T)-np.exp(-3*t))+ 4/25*(np.exp(-1.5*T)-np.exp(-1.5*t))
    
    d2= (np.log(S/K) + r*(T-t)-1/2*v)/np.sqrt(v)
    
    d1= d2+np.sqrt(v)
    
    V=K*np.exp(-r*(T-t))*st.norm.cdf(-d2)- S*st.norm.cdf(-d1)
    
    return V

def GMBPaths(r, T, t, NoOfPaths, NoOfSteps, S0):
    ## put option value calculated using simulations with the time-dependent sigma

    time=np.zeros([NoOfSteps+1])
    sigma_t=np.zeros([NoOfSteps+1])
    X=np.zeros([NoOfPaths, NoOfSteps+1])
    Z=np.random.normal(0,1,[NoOfPaths, NoOfSteps])
    
    dt=T/NoOfSteps
    
    X[:,0]= np.log(S0)
    
    for i in range(0, NoOfSteps):
        
        if(NoOfPaths>1):
            Z[:,i]=(Z[:,i]-np.mean(Z[:,i]))/np.std(Z[:,i])  
        
        sigma_t[i]=0.6-0.2*np.exp(-1.5*time[i])
        X[:,i+1]=X[:,i]+ (r-1/2*np.power(sigma_t[i],2))*dt+ sigma_t[i]*np.sqrt(dt)*Z[:,i]
        time[i+1]=time[i]+dt
           
    S=np.exp(X)

    ST=S[:,-1] 
    
    return ST


def BS_PutOptionValue(K, r, tau, sigmaBS, S0):
    ## put option value calculated using Black Scholes formula
    
   
    d2= (np.log(S0/float(K)) + (r-1/2*sigmaBS**2)*tau)/(sigmaBS*np.sqrt(tau))
    
    d1= d2+sigmaBS*np.sqrt(tau)
    
    V=K*np.exp(-r*tau)*st.norm.cdf(-d2)- S0*st.norm.cdf(-d1)
    
    return V

def dV_dsigma(S0, K, sigma, tau, r):
    
    d2= (np.log(S0/float(K))+(r-1/2*sigma**2)*tau)/(sigma*np.sqrt(tau))
    vega=K* np.exp(-r*tau)*st.norm.pdf(d2)*np.sqrt(tau)

    return vega 

def ImpliedVolatility( K, r, T, sigma, S0, V_Market):
    
    error = 1 # initial error
    
    optPrice= lambda sigma: BS_PutOptionValue(K, r, tau, sigma, S0)
    vega= lambda sigma: dV_dsigma(S0, K, sigma, tau, r)

    while error>1e-6:

        g= V_Market- optPrice(sigma)
        g_prim = -vega(sigma)
        sigma_new=sigma - g /g_prim
        
        #error=abs(sigma_new-sigma)
        error=abs(g)
        sigma=sigma_new
        
       # print('Error = {0}'. format(error))
        
    return sigma



    

T=5
t=0

tau=T-t
S0=1 
r=0.05

K=1.6
sigmaInit=0.4
    
NoOfSteps=1000
NoOfPaths=200000

V_MarketAnalytic=Market_PutOptionValueAnalytic(K, r, T, S0, t )
ST=GMBPaths(r, T, t, NoOfPaths, NoOfSteps, S0)   

PayOff=np.maximum((K-ST),0)
V_MarketNumeric=np.exp(-r*(T-t))*np.mean(PayOff) 

print("VMarket=", V_MarketNumeric)
print("VAnalytic=",V_MarketAnalytic)

K_values=np.linspace(0.5, 2, 21)
ImpliedVol=np.zeros(len(K_values))
 
for i in range(len(K_values)):
    
    PayOff=np.maximum((K_values[i]-ST),0)
    V_MarketNumeric=np.exp(-r*(T-t))*np.mean(PayOff)    
    ImpliedVol[i] =ImpliedVolatility( K_values[i], r, T, sigmaInit, S0, V_MarketNumeric)

plt.figure()
plt.plot(K_values, ImpliedVol*100)
plt.xlabel("strike(K)")
plt.ylabel("Implied volatility (%)")
plt.ylim([55, 60])




