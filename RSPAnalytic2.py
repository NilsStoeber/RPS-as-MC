# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from scipy.linalg import null_space

a = [1,1,1]    # Auszahlungsparameter: Verlust
b = [1,1,1]    # Auszahlungsparameter: Gewinn
eps = 0.1      # Zufallsparameter (Niedrig = Niedriger Zufallsfaktor, Hoch = Hoher Zufallsfaktor)
N = 50         # Populationsgröße

# Funktionen für erwartete Auszahlung und die gegebene Update-Regel
def updatepi(f, N):
    return [(b[2]*(N-sum(f))-a[1]*f[1])/N, (b[0]*f[0]-a[2]*(N-sum(f)))/N, (b[1]*f[1]-a[0]*f[0])/N]

def T(N,f,k,l):
    pi = updatepi(f,N)
    return eps + max(pi[l]-pi[k],0)

# Erstellung der Übergangsmatrix
states = (N+1)*(N+2)//2
P = np.zeros((states,states))

for i in range(N+1):
    for j in range(i+1):
        k = (i+1)*(i+2)//2-i-1+j
        f = [j,N-i]
        if j != 0:          # Wechsel von Papier weg ist möglich
            P[k,k-1] = T(N,f,0,2)
            P[k,k-i-1] = T(N,f,0,1)
        if i != j:          # Wechsel von Stein weg ist möglich
            P[k,k+1] = T(N,f,2,0)
            P[k,k-i] = T(N,f,2,1)
        if i != N:          # Wechsel von Schere weg ist möglich
            P[k,k+i+1] = T(N,f,1,2)
            P[k,k+i+2] = T(N,f,1,0)
        P[k,:] = P[k,:]/sum(P[k,:])

Sa = [0,states-N-1,states-1]
St = np.hstack((np.arange(1,states-N-1), np.arange(states-N,states-1)))
I = np.take(np.take(P, Sa, 0), Sa, 1)
R = np.take(np.take(P, St, 0), Sa, 1)
R2 = np.take(np.take(P,Sa,0), St,1)
Q = np.take(np.take(P, St, 0), St, 1)
P = np.vstack((np.hstack((np.zeros((3,3)),R2)),np.hstack((R,Q))))

# Berechnung des Eigenvektor von P bzw. der stationären Verteilung
w = null_space(-P.T+np.identity(states))
w = w/sum(w)


#%% Extrahiert die Zustände mit f_1 = N-x
def pList(pC, x):
    if x == 0:
        return [pC[0]]
    if x == N:
        L = [pC[1]]
        for i in range(len(pC)-N+1, len(pC)):
            L.append(pC[i])
        L.append(pC[2])
        return L
   
    L2 = []
    for i in range((x+1)*(x+2)//2-x+1,((x+1)*(x+2)//2+2)):
        L2.append(pC[i])
    return L2 


#%% Plotten der stationären Verteilung

ER = np.zeros((N+1, 2*N+1))
for i in range(N+1):
    for j in range(2*N+1):
        ER[i,j] = 'nan'
for i in range(N+1):
    for j in range(0,i+1):
        ER[i,N-i+2*j] = pList(w,i)[j]
for i in range(N+1):
    for j in range(0,i):
        ER[i,N-i+2*j+1] = (ER[i,N-i+2*j]+ER[i,N-i+2*j+2])/2

sb.heatmap(ER, cmap = "Spectral", yticklabels = False, xticklabels = False)
plt.text(-0.07*N,1.05*N,"R")
plt.text(N, -0.03*N, "S")
plt.text(2.07*N,1.05*N,"P")
plt.show()