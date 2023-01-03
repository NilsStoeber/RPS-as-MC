# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import random as r
import seaborn as sb

a = [1,1,1]    # Auszahlungsparameter: Verlust
b = [1,1,1]    # Auszahlungsparameter: Gewinn
eps = 0.1      # Zufallsparameter (Niedrig = Niedriger Zufallsfaktor, Hoch = Hoher Zufallsfaktor)
N = 50         # Populationsgröße

it = 100000000 # Iterationsschritte

# Funktionen für erwartete Auszahlung und die gegebene Update-Regel
def updatepi(f, N):
    return [(b[2]*(N-sum(f))-a[1]*f[1])/N, (b[0]*f[0]-a[2]*(N-sum(f)))/N, (b[1]*f[1]-a[0]*f[0])/N]

def p(N,f,pi,k,l):
    f2 = [f[0],f[1],N-sum(f)]
    if f2[k] == 0 or f2[l] == N:
        return 0
    else: return max(pi[l]-pi[k],0)+eps


# Erstellung von Matrizen mit den Übergangswahrscheinlichkeiten vom gegebenen Zustand
def createP(N, f):
    pi = updatepi(f,N)
    P = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            if i != j:
                tp = p(N,f,pi,i,j)
                P[i,j] = tp
    P = P/np.sum(P)
    return P

def createAll(N):
    m = []
    for i in range(N+1):
        m.append([])
        for j in range(i+1):
            m[i].append(createP(N,[N-i,j]))
    return m

Ps = createAll(N)


# Simuliert das evolutionäre RPS-Spiel ausgehend vom gegebenen Startzustand f
# f = [f_1, f_2] für absolute Häufigkeiten von Papier und Schere
def simulation(N = N, fParam = [0,0]):   
    duration = 0
    f = np.copy(fParam)
    A = np.zeros((N+1,N+1))
    while duration < it:
        duration+=1
        sample = r.random()
        
        P = Ps[N-f[0]][f[1]]
        # Samplen des Strategiewechsels
        if sample < P[0,1]:                                             # Papier zu Schere
            f[0] -= 1
            f[1] += 1
        elif sample < sum((P[0,1],P[0,2])):                             # Papier zu Stein
            f[0] -= 1
        elif sample < sum((P[0,1],P[0,2],P[1,0])):                      # Schere zu Papier
            f[0] += 1   
            f[1] -= 1
        elif sample < sum((P[0,1],P[0,2],P[1,0],P[2,0])):               # Stein zu Papier
            f[0] += 1
        elif sample < sum((P[0,1],P[0,2],P[1,0],P[2,0],P[1,2])):        # Schere zu Stein
            f[1] -= 1
        elif sample < sum((P[0,1],P[0,2],P[1,0],P[2,0],P[1,2],P[2,1])): # Stein zu Schere
            f[1] += 1
        # Ansonsten kein Strategiewechsel, Zustand bleibt gleich
         
        # Nutze jeden 1000ten Schritt für die Approximation der
        # stationären Verteilung (kann variiert werden)
        steps = 1000
        if (duration % steps) == 0: 
            A[f[0],f[1]] += 1
    
    return(f[0],f[1], A/it/steps)

#%% Wähle einen zufälligen Startzustand

start = np.random.randint(0,(N+1)*(N+2)//2)
a = 0
f = [0,0]
for i in range(N+1):
    if (i*(i+1)//2) > start:
        a = i-1
        break
    a = N
f[0] = start-a*(a+1)//2
f[1] = N-a

#%% Start der Simulation und Plotten der Resultate

A = simulation(N,f)[-1]
ER = np.zeros((N+1, 2*N+1))
for i in range(N+1):
    for j in range(2*N+1):
        ER[i,j] = 'nan'
for i in range(N+1):
    for j in range(0,i+1):
        ER[i,N-i+2*j] = A[j,N-i]
for i in range(N+1):
    for j in range(0,i):
        ER[i,N-i+2*j+1] = (ER[i,N-i+2*j]+ER[i,N-i+2*j+2])/2
        
sb.heatmap(ER, cmap = "Spectral", yticklabels = False, xticklabels = False)
plt.text(-0.07*N,1.05*N,"R")
plt.text(N, -0.03*N, "S")
plt.text(2.07*N,1.05*N,"P")
plt.show()