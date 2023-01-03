# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

a = [1,1,1]   # Auszahlungsparameter: Verlust
b = [1,1,1]   # Auszahlungsparameter: Gewinn
beta = 0.1    # Zufallsparameter (Niedrig = Hoher Zufallsfaktor, Hoch = Niedriger Zufallsfaktor)
N = 50        # Populationsgröße

# Funktionen für erwartete Auszahlung und die gegebene Update-Regel
def F(x,y):
    return 1/(1 + np.e**(beta*(x-y)))

def updatepi(f, N):
    return [(b[2]*(N-sum(f))-a[1]*f[1])/(N-1), (b[0]*f[0]-a[2]*(N-sum(f)))/(N-1), (b[1]*f[1]-a[0]*f[0])/(N-1)]

def p(N,f,k,l):
    fp = [f[0],f[1],N-f[0]-f[1]]
    pi = updatepi(f,N)
    return 2*fp[k]*fp[l]/N*F(pi[k],pi[l])/(N-1)


# Erstellung der Übergangsmatrix
states = (N+1)*(N+2)//2
P = np.zeros((states,states))

for i in range(N+1):
    for j in range(i+1):
        k = (i+1)*(i+2)//2-i-1+j
        f = [j,N-i]
        if j != 0 and i != j:       # Wechsel zwischen Papier und Stein möglich
            P[k,k-1] = p(N,f,0,2)
            P[k,k+1] = p(N,f,2,0)
        if i != N and i != j:       # Wechsel zwischen Schere und Stein möglich
            P[k,k+i+1] = p(N,f,1,2)
            P[k,k-i] = p(N,f,2,1)
        if i != N and j != 0:       # Wechsel zwischen Papier und Schere möglich
            P[k,k+i+2] = p(N,f,1,0)
            P[k,k-i-1] = p(N,f,0,1)
        P[k,k] = 1 - sum(P[k,:])    # Wahrscheinlichkeit Empty-Step


# Umformen von P in die kanonische Form
Sa = [0,states-N-1,states-1]
St = np.hstack((np.arange(1,states-N-1), np.arange(states-N,states-1)))
I = np.take(np.take(P, Sa, 0), Sa, 1)
R = np.take(np.take(P, St, 0), Sa, 1)
Q = np.take(np.take(P, St, 0), St, 1)
P = np.vstack((np.hstack((Q,R)),np.hstack((np.zeros((3,Q.shape[0])),I))))


#%% Fundamentalmatrix und Resultate der Fundamentalmatrix-Methode
Nmat = np.linalg.inv(np.identity(Q.shape[0])-Q)
c = np.ones(Q.shape[0])
t = Nmat@c
B = Nmat@R

Na = [np.zeros_like(Nmat) for i in range(3)]
for i in range(Nmat.shape[0]):
    for j in range(Nmat.shape[1]):
        if B[i,0] > 0:
            Na[0][i,j] = B[j,0]/B[i,0]*Nmat[i,j]
        else: Na[0][i,j] = 'nan'
        if B[i,1] > 0:
            Na[1][i,j] = B[j,1]/B[i,1]*Nmat[i,j]
        else: Na[1][i,j] = 'nan'
        if B[i,2] > 0:
            Na[2][i,j] = B[j,2]/B[i,2]*Nmat[i,j]
        else: Na[2][i,j] = 'nan'
        
ta = [Na[i]@c for i in range(3)]


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


#%% Absorptionswahrscheinlichkeiten von Papier

pP = np.hstack((I[:,2],B[:,2]))
ER = np.zeros((N+1, 2*N+1))
for i in range(N+1):
    for j in range(2*N+1):
        ER[i,j] = 'nan'
for i in range(N+1):
    L = pList(pP,i)
    for j in range(0,i+1):
        ER[i,N-i+2*j] = L[j]
for i in range(N+1):
    for j in range(0,i):
        ER[i,N-i+2*j+1] = (ER[i,N-i+2*j]+ER[i,N-i+2*j+2])/2

sb.heatmap(ER, cmap = "Spectral", yticklabels = False, xticklabels = False)
plt.text(-0.07*N,1.05*N,"R")
plt.text(N, -0.03*N, "S")
plt.text(2.07*N,1.05*N,"P")
plt.show()


#%% Absorptionswahrscheinlichkeiten von Schere

pS = np.hstack((I[:,0],B[:,0]))
ER = np.zeros((N+1, 2*N+1))
for i in range(N+1):
    for j in range(2*N+1):
        ER[i,j] = 'nan'
for i in range(N+1):
    L = pList(pS,i)
    for j in range(0,i+1):
        ER[i,N-i+2*j] = L[j]
for i in range(N+1):
    for j in range(0,i):
        ER[i,N-i+2*j+1] = (ER[i,N-i+2*j]+ER[i,N-i+2*j+2])/2

sb.heatmap(ER, cmap = "Spectral", yticklabels = False, xticklabels = False)
plt.text(-0.07*N,1.05*N,"R")
plt.text(N, -0.03*N, "S")
plt.text(2.07*N,1.05*N,"P")
plt.show()


#%% Absorptionswahrscheinlichkeiten von Stein

pR = np.hstack((I[:,1],B[:,1]))

ER = np.zeros((N+1, 2*N+1))
for i in range(N+1):
    for j in range(2*N+1):
        ER[i,j] = 'nan'
for i in range(N+1):
    L = pList(pR,i)
    for j in range(0,i+1):
        ER[i,N-i+2*j] = L[j]
for i in range(N+1):
    for j in range(0,i):
        ER[i,N-i+2*j+1] = (ER[i,N-i+2*j]+ER[i,N-i+2*j+2])/2

sb.heatmap(ER, cmap = "Spectral", yticklabels = False, xticklabels = False)
plt.text(-0.07*N,1.05*N,"R")
plt.text(N, -0.03*N, "S")
plt.text(2.07*N,1.05*N,"P")
plt.show()


#%% Unbedingte Absorptionszeiten

tGen = np.hstack((np.zeros(3),t))
ER = np.zeros((N+1, 2*N+1))
for i in range(N+1):
    for j in range(2*N+1):
        ER[i,j] = 'nan'
for i in range(N+1):
    L = pList(tGen,i)
    for j in range(0,i+1):
        ER[i,N-i+2*j] = L[j]
for i in range(N+1):
    for j in range(0,i):
        ER[i,N-i+2*j+1] = (ER[i,N-i+2*j]+ER[i,N-i+2*j+2])/2

sb.heatmap(ER, cmap = "Spectral", yticklabels = False, xticklabels = False, vmin = 0, vmax = 1600)
plt.text(-0.07*N,1.05*N,"R")
plt.text(N, -0.03*N, "S")
plt.text(2.07*N,1.05*N,"P")
plt.show()


#%% Absorptionszeiten von Papier

tP = np.hstack((['nan','nan',0],ta[2]))
ER = np.zeros((N+1, 2*N+1))
for i in range(N+1):
    for j in range(2*N+1):
        ER[i,j] = 'nan'
for i in range(N+1):
    L = pList(tP,i)
    for j in range(0,i+1):
        ER[i,N-i+2*j] = L[j]
for i in range(N+1):
    for j in range(0,i):
        ER[i,N-i+2*j+1] = (ER[i,N-i+2*j]+ER[i,N-i+2*j+2])/2

sb.heatmap(ER, cmap = "Spectral", yticklabels = False, xticklabels = False)
plt.text(-0.07*N,1.05*N,"R")
plt.text(N, -0.03*N, "S")
plt.text(2.07*N,1.05*N,"P")
plt.show()


#%% Absorptionszeiten von Schere

tS = np.hstack(([0,'nan','nan'],ta[0]))
ER = np.zeros((N+1, 2*N+1))
for i in range(N+1):
    for j in range(2*N+1):
        ER[i,j] = 'nan'
for i in range(N+1):
    L = pList(tS,i)
    for j in range(0,i+1):
        ER[i,N-i+2*j] = L[j]
for i in range(N+1):
    for j in range(0,i):
        ER[i,N-i+2*j+1] = (ER[i,N-i+2*j]+ER[i,N-i+2*j+2])/2

sb.heatmap(ER, cmap = "Spectral", yticklabels = False, xticklabels = False)
plt.text(-0.07*N,1.05*N,"R")
plt.text(N, -0.03*N, "S")
plt.text(2.07*N,1.05*N,"P")
plt.show()


#%% Absorptionszeiten von Stein

tR = np.hstack((['nan',0,'nan'],ta[1]))
ER = np.zeros((N+1, 2*N+1))
for i in range(N+1):
    for j in range(2*N+1):
        ER[i,j] = 'nan'
for i in range(N+1):
    L = pList(tR,i)
    for j in range(0,i+1):
        ER[i,N-i+2*j] = L[j]
for i in range(N+1):
    for j in range(0,i):
        ER[i,N-i+2*j+1] = (ER[i,N-i+2*j]+ER[i,N-i+2*j+2])/2

sb.heatmap(ER, cmap = "Spectral", yticklabels = False, xticklabels = False)
plt.text(-0.07*N,1.05*N,"R")
plt.text(N, -0.03*N, "S")
plt.text(2.07*N,1.05*N,"P")
plt.show()