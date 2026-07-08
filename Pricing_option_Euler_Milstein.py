# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 17:00:15 2026

@author: sbanerj1
"""

import numpy as np
import scipy.stats as st

def CallOptionValue(NoOfPaths, NoOfSteps, T, S0, K):
    
    Z=np.random.normal(0,1,[NoOfPaths, NoOfSteps])
    SE=np.zeros([NoOfPaths, NoOfSteps+1])
    SM=np.zeros([NoOfPaths, NoOfSteps+1])
    sigma=np.zeros([NoOfSteps+1])
    time=np.zeros([NoOfSteps+1])
    r=np.zeros([NoOfSteps+1])
    
    SE[:,0]=S0
    SM[:,0]=S0
    
    dt=T/NoOfSteps
    
    
    for i in range(0, NoOfSteps):
        
        if NoOfPaths>1:
            Z[:,i]=(Z[:,i]-np. mean(Z[:,i]))/np.std(Z[:,i])
        
        sigma[i]=0.6-0.2*np.exp(-1.5*time[i])
        
        if time[i]<2:
            r[i]=0.05
        else:
            r[i]=0.08
       
       # Euler maruyama 
       
        SE[:,i+1]=SE[:,i]+r[i]*SE[:,i]*dt+sigma[i]*SE[:,i]*np.sqrt(dt)*Z[:,i]
            
       # Milstein 
        SM[:,i+1]=SM[:,i]+r[i]*SM[:,i]*dt+sigma[i]*SM[:,i]*np.sqrt(dt)*Z[:,i]+0.5*np.pow(sigma[i],2)*SM[:,i]*(dt*np.pow(Z[:,i],2)-dt)
        
        time[i+1]=time[i]+dt    
        
   # discount=Integration|_0^T e^(-r(t)dt)
    discount=np.exp(-0.26)
    
    payoffSE=discount*np.maximum((SE[:,-1])-K,0)
    payoffSM=discount*np.maximum((SM[:,-1])-K,0)
    
    return SE, np.mean(payoffSE), SM, np.mean(payoffSM)

def CallOptionValueAnalytic(K, T, S, t ):
    ## put option value calculated analytically from the GBM with time-dependent sigma

    v=0.36*(T-t)-0.04/3*(np.exp(-3*T)-np.exp(-3*t))+ 4/25*(np.exp(-1.5*T)-np.exp(-1.5*t))
    
    d2= (np.log(S/K) + 0.26-1/2*v)/np.sqrt(v)
    
    d1= d2+np.sqrt(v)
    
    V=S*st.norm.cdf(d1)-K*np.exp(-0.26)*st.norm.cdf(d2) 
    
    return V

def BarrierPayOff(S, B, K):
    
    payoff=np.exp(-0.26)*np.maximum((S[:,-1]-K),0)
    
    Smax= np.max(S, axis=1)
    
    out_indicator=(Smax<B)
    
    payoff_UO=payoff*out_indicator  
    
    in_indicator=(Smax>=B)
    
    payoff_UI=payoff*in_indicator
    
    return(np.mean(payoff_UI), np.mean(payoff_UO))
    
     
    
K=1.6
T=4
S0=1
t=0

B=2.5

V=CallOptionValueAnalytic(K, T, S0, t )
print("The analytical value of the call option=", V)   
       
    
NoOfSteps= [50, 100, 200, 400]
NoOfPaths= [1000, 10000, 100000]
    
PriceE=np.zeros([len(NoOfPaths), len(NoOfSteps)])
PriceM=np.zeros([len(NoOfPaths), len(NoOfSteps)])

ErrorE=np.zeros([len(NoOfPaths), len(NoOfSteps)])
ErrorM=np.zeros([len(NoOfPaths), len(NoOfSteps)])

PriceBarrierE_UI=np.zeros([len(NoOfPaths), len(NoOfSteps)])
PriceBarrierE_UO=np.zeros([len(NoOfPaths), len(NoOfSteps)])

PriceBarrierM_UI=np.zeros([len(NoOfPaths), len(NoOfSteps)])
PriceBarrierM_UO=np.zeros([len(NoOfPaths), len(NoOfSteps)])

    
for i in range (len(NoOfPaths)):
        
    
        
    for j in range(len(NoOfSteps)):
        
            
        SE, PriceE[i,j], SM, PriceM[i,j] =CallOptionValue(NoOfPaths[i], NoOfSteps[j], T, S0, K)
        ErrorE[i,j] = PriceE[i,j]-V
        ErrorM[i,j] = PriceM[i,j]-V
        
        PriceBarrierE_UI[i,j], PriceBarrierE_UO[i,j] =BarrierPayOff(SE, B, K)
        PriceBarrierM_UI[i,j], PriceBarrierM_UO[i,j] =BarrierPayOff(SM, B, K)
            
            
    
print("No of paths:", NoOfPaths[-1],"No of Steps:", NoOfSteps[-1])
print("Price (Euler):", PriceE[-1,-1] )
print("Error (Euler):", ErrorE[-1,-1])
print("Price (Milstein):", PriceM[-1,-1] )
print("Error (Milstein):", ErrorM[-1,-1])
print("Up-and-in barrier option (Euler):", PriceBarrierE_UI[-1,-1] )
print("Up-and-out barrier option (Euler):", PriceBarrierE_UO[-1,-1])
print("Up-and-in barrier option (Milstein):", PriceBarrierM_UI[-1,-1] )
print("Up-and-out barrier option (Milstein):", PriceBarrierM_UO[-1,-1])


    
    
        
        
        
        
        
    