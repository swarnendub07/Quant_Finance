# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 00:25:18 2026

@author: sbanerj1
"""

import numpy as np

def GBMpaths(r, T, sigma1, sigma2, S01, S02, rho, NoOfPaths, NoOfSteps):
    
    time=np.zeros([NoOfSteps+1])
    Z1=np.random.normal(0,1,[NoOfPaths, NoOfSteps])
    Z2=np.random.normal(0,1,[NoOfPaths, NoOfSteps])
    S1=np.zeros([NoOfPaths, NoOfSteps+1])
    S2=np.zeros([NoOfPaths, NoOfSteps+1])
    
    S1[:,0]=S01;
    S2[:,0]=S02;
    
    dt=T/NoOfSteps
    
    for i in range(0, NoOfSteps):
        
        if(NoOfPaths>1):
            Z1[:,i]=(Z1[:,i]-np.mean(Z1[:,i]))/np.std(Z1[:,i])
            Z2[:,i]=(Z2[:,i]-np.mean(Z2[:,i]))/np.std(Z2[:,i])
            
        
        S1[:,i+1]=S1[:,i]+r*S1[:,i]*dt+ sigma1*S1[:,i]*np.sqrt(dt)*Z1[:,i]
        S2[:,i+1]=S2[:,i]+r*S2[:,i]*dt+ sigma2*S2[:,i]*(rho*np.sqrt(dt)*Z1[:,i]+np.sqrt(1-rho**2)*np.sqrt(dt)*Z2[:,i])
    
        time[i+1]=time[i]+dt
        
    PayOff=S1[:,-1]*np.maximum(S1[:,-1],S2[:,-1])
    
    discount=np.exp(-r*T)
    
    Price=discount*PayOff
    
    return (S1, S2, np.mean(PayOff), np.mean(Price))

    
    
def mainCalculation():
    
    r=0.01
    S01=1
    S02=1
    T=5
    sigma1=0.4
    sigma2=0.15
    
    NoOfPaths=1000
    NoOfSteps=1000
    
    rho=-0.9
    S1_rhoPos, S2_rhoPos, PayOff_rhoPos, Price_rhoPos=GBMpaths(r, T, sigma1, sigma2, S01, S02, rho, NoOfPaths, NoOfSteps)
    
    rho=0.9
    S1_rhoNeg, S2_rhoNeg, PayOff_rhoNeg, Price_rhoNeg=GBMpaths(r, T, sigma1, sigma2, S01, S02, rho, NoOfPaths, NoOfSteps)
    
    print(Price_rhoPos)
    print(Price_rhoNeg)
    
mainCalculation()   