# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

a = [1,1,1]   # Auszahlungsparameter: Verlust
b = [1,1,1]   # Auszahlungsparameter: Gewinn
beta = 0.5    # Zufallsparameter Modell 1
eps = 0.1     # Zufallsparameter Modell 2
N = 15        # Populationsgröße

# Funktionen für erwartete Auszahlung und die gegebenen Update-Regeln
def F(x,y):
    return 1/(1 + np.e**(beta*(x-y)))

def updatepi(f, N):
    return [(b[2]*(N-sum(f))-a[1]*f[1])/(N-1), (b[0]*f[0]-a[2]*(N-sum(f)))/(N-1), (b[1]*f[1]-a[0]*f[0])/(N-1)]

def p(N,f,pi,k,l):
    fp = [f[0],f[1],N-f[0]-f[1]]
    return 2*fp[k]*fp[l]/N*F(pi[k],pi[l])/(N-1)

def T(N,f,k,l):
    pi = updatepi(f,N)
    return eps + max(pi[l]-pi[k],0)


# Erstellung von Matrizen mit den Übergangswahrscheinlichkeiten vom gegebenen Zustand
def createP(N, f):  # Modell 1
    pi = updatepi(f,N)
    P = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            if i != j:
                tp = p(N,f,pi,i,j)
                P[i,j] = tp
    return P

def createP2(N, f): # Modell 2
    P = np.zeros((3,3))
    fUp = [f[0],f[1],N-f[0]-f[1]]
    for i in range(3):
        for j in range(3):
            if i != j and fUp[i]!=0:
                tp = T(N,f,i,j)
                P[i,j] = tp
    P = P/np.sum(P)                
    return P


#%% Plotten des Vektorfeldes der gewichteten Übergangswahrscheinlichkeiten
# vecscale zur Anpassung der Vektorengröße, je nach Parameterwahl
# method zur Wahl des Modells 1 oder 2
def plotVF(vecscale = 3, method = 1):
    # Rahmen
    plt.rcParams["figure.figsize"] = (8,8)
    plt.plot((N/2*np.sqrt(5/4),N*np.sqrt(5/4)),(N,0), c = "black", alpha = 0.5)
    plt.plot((0,N/2*np.sqrt(5/4)),(0,N), c = "black", alpha = 0.5)
    plt.plot((0,N*np.sqrt(5/4)),(0,0), c = "black", alpha = 0.5)
    
    x,y = np.meshgrid(np.linspace(0,N*np.sqrt(5/4),2*N+1),np.linspace(0,N,N+1))
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    for i in range(N+1):
        for j in range(2*N+1):
            u[i,j] = 'nan'
            v[i,j] = 'nan'
    for i in range(N+1): # Schere = N-i
        for j in range(0,i+1): # Papier = j
            f = [j,N-i]
            if method == 1: 
                P = createP(N,f)
            elif method == 2:
                P = createP2(N,f)
            # Berechnung des Vektors
            dx = (P[2,1]-P[1,2]) + (P[1,0]-P[0,1]) + 2*(P[2,0] - P[0,2])
            dy = (P[2,1]-P[1,2]) + (P[0,1]-P[1,0])
            u[N-i,N-i+2*j] = dx/2*np.sqrt(5/4)
            v[N-i,N-i+2*j] = dy

    plt.axis('off')
    plt.axis('equal')
    plt.text(N*1.02*np.sqrt(5/4),0,"P")
    plt.text(-0.05*N,0,"R")
    plt.quiver(x,y,u,v, scale = vecscale)
    plt.yticks()
    plt.text(N*0.485*np.sqrt(5/4),N*1.02,"S")
    plt.show()

#%%
plotVF(2, method = 1)
plotVF(18, method = 2)
