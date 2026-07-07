# -*- coding: utf-8 -*-
"""
Created on Sat May 30 09:12:38 2026

Exercise 2 Question 1
@author: sbanerj1
"""

## dS(t)=rS(t)dt+sigma(t)S(t)dW(t)^Q
## sigma(t)=0.6-0.2e^(-1.5t)

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

    
    
    return S


def BS_PutOptionValue(K, r, tau, sigmaBS, S0):
    ## put option value calculated using Black Scholes formula
   
    d2= (np.log(S0/float(K)) + (r-1/2*sigmaBS**2)*tau)/(sigmaBS*np.sqrt(tau))
    
    d1= d2+sigmaBS*np.sqrt(tau)
    
    V=K*np.exp(-r*tau)*st.norm.cdf(-d2)- S0*st.norm.cdf(-d1)
    
    return V

def dV_dsigma(S0, K, sigma, tau, r):
    ## vega which is used in Newton-Raphson method to find Impied Volatility
    
    d2= (np.log(S0/float(K))+(r-1/2*sigma**2)*tau)/(sigma*np.sqrt(tau))
    vega=K* np.exp(-r*tau)*st.norm.pdf(d2)*np.sqrt(tau)

    return vega 

def ImpliedVolatility( K, r, T, sigma, S0, V_Market):
    ## Calculate implied volatility
    
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



# Parameters    

T=6
t=0
tau=T-t
S0=1 
r=0.05
K=1.7
sigmaInit=0.4
    
NoOfSteps=6000
NoOfPaths=10000
dt=T/NoOfSteps


# Calculate Put option prices analytically and using Monte Carlo simulations
# for time-dependent volatility
V_MarketAnalytic=Market_PutOptionValueAnalytic(K, r, T, S0, t )
S=GMBPaths(r, T, t, NoOfPaths, NoOfSteps, S0)   
ST=S[:,-1]

PayOff=np.maximum((K-ST),0)
V_MarketNumeric=np.exp(-r*(T-t))*np.mean(PayOff) 

print("VMarket=", V_MarketNumeric)
print("VAnalytic=",V_MarketAnalytic)


# Analyze the implied volatility when market prices for Put option are calculated using Monte Carlo simulations
# and analytically
K_values=np.linspace(0.5, 2, 21)
ImpliedVol=np.zeros(len(K_values))
ImpliedVol_A=np.zeros(len(K_values))
 
for i in range(len(K_values)):
    
     
    PayOff=np.maximum((K_values[i]-ST),0)
    V_MarketNumeric=np.exp(-r*(T-t))*np.mean(PayOff)    
    ImpliedVol[i] =ImpliedVolatility( K_values[i], r, T, sigmaInit, S0, V_MarketNumeric)
    
    V_MarketAnalytic=Market_PutOptionValueAnalytic(K_values[i], r, T, S0, t )
    ImpliedVol_A[i] =ImpliedVolatility( K_values[i], r, T, sigmaInit, S0, V_MarketAnalytic)

plt.figure()
plt.plot(K_values, ImpliedVol*100)
plt.title("Market data from Monte Carlo simulations")
plt.xlabel("strike(K)")
plt.ylabel("Implied volatility (%)")
plt.ylim([55, 60])

plt.figure()
plt.plot(K_values, ImpliedVol_A*100)
plt.title("Market data from Analytic expression")
plt.xlabel("strike(K)")
plt.ylabel("Implied volatility (%)")
plt.ylim([55, 60])


# Analyze the implied volatility for different expires when
# market prices for Put option are calculated using Monte Carlo simulations
# and analytically

T_values=np.linspace(1, 6, 21)
ImpliedVol_TK=np.zeros([len(T_values), len(K_values)])
ImpliedVol_A_TK=np.zeros([len(T_values), len(K_values)])

for i in range(len(T_values)):
    
    Sindex=int(T_values[i]/dt)
    ST=S[:,Sindex]
    
    print(Sindex)
    
    for j in range(len(K_values)):
        
        PayOff=np.maximum((K_values[j]-ST),0)
        V_MarketNumeric=np.exp(-r*(T_values[i]-t))*np.mean(PayOff)    
        ImpliedVol_TK[i,j] =ImpliedVolatility( K_values[j], r, T_values[i], sigmaInit, S0, V_MarketNumeric)
        
        V_MarketAnalytic=Market_PutOptionValueAnalytic(K_values[j], r, T_values[i], S0, t )
        ImpliedVol_A_TK[i,j] =ImpliedVolatility( K_values[j], r, T_values[i], sigmaInit, S0, V_MarketAnalytic)


# plot T vs implied volatility        
j=16 # K=1.7

plt.figure()
plt.plot(T_values, ImpliedVol_TK[:,16]*100)
plt.title("Market data from Monte Carlo simulations")
plt.xlabel("Time to maturity (T)")
plt.ylabel("Implied volatility (%)")
plt.ylim([50, 60])
plt.show()

plt.figure()
plt.plot(T_values, ImpliedVol_A_TK[:,16]*100)
plt.title("Market data from Analytic expression")
plt.xlabel("Time to maturity (T)")
plt.ylabel("Implied volatility (%)")
plt.ylim([50, 60])
plt.show()


# plot T vs K vs implied volatility in a 3d plot  
K_grid, T_grid = np.meshgrid(K_values, T_values)

# 3D plot
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

surface = ax.plot_surface(K_grid, T_grid, ImpliedVol_TK*100, cmap='viridis')
ax.set_title("Market data from Monte Carlo simulations")
ax.set_xlabel("Strike K")
ax.set_ylabel("Maturity T")
ax.set_zlabel("Implied Volatility (%)")

fig.colorbar(surface, shrink=0.5)
plt.show()

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

surface = ax.plot_surface(K_grid, T_grid, ImpliedVol_A_TK*100, cmap='viridis')
ax.set_title("Market data from Analytic expression")
ax.set_xlabel("Strike K")
ax.set_ylabel("Maturity T")
ax.set_zlabel("Implied Volatility (%)")

fig.colorbar(surface, shrink=0.5)

plt.show()


