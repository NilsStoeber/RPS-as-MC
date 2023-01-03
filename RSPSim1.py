# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import random as r
import seaborn as sb

a = [1,1,1]   # Auszahlungsparameter: Verlust
b = [1,1,1]   # Auszahlungsparameter: Gewinn
beta = 0.1    # Zufallsparameter (Niedrig = Hoher Zufallsfaktor, Hoch = Niedriger Zufallsfaktor)
N = 50        # Populationsgröße

t = 500     # Anzahl Simulationen pro Zustand

# Funktionen für erwartete Auszahlung und die gegebene Update-Regel
def F(x,y):
    return 1/(1 + np.e**(beta*(x-y)))

def updatepi(f, N):  
    return [(b[2]*(N-sum(f))-a[1]*f[1])/(N-1), (b[0]*f[0]-a[2]*(N-sum(f)))/(N-1), (b[1]*f[1]-a[0]*f[0])/(N-1)]

def p(N,f,pi,k,l):   
    fp = [f[0],f[1],N-f[0]-f[1]]
    return 2*fp[k]*fp[l]/N*F(pi[k],pi[l])/(N-1)


# Erstellung von Matrizen mit den Übergangswahrscheinlichkeiten vom gegebenen Zustand
def createP(N, f):
    pi = updatepi(f,N)
    P = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            if i != j:
                tp = p(N,f,pi,i,j)
                P[i,j] = tp                
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
    
    # Das Spiel läuft, solange noch mehr als eine Strategie existiert
    while max(f) < N and sum(f) > 0:
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

    return(f[0],f[1],N-sum(f),duration)

#%% t-fache Simulation des gegebenen Startzustands
def simulateSpot(N, f, t):
    p = 0
    s = 0
    r = 0
    timeP = 0
    timeS = 0
    timeR = 0
    # Zähle Absorptionen und die zugehörigen Zeiten pro Strategie
    for k in range(t):
        res = simulation(N,f)
        if res[0] == N:
            p += 1
            timeP += res[3]
        elif res[1] == N:
            s+= 1
            timeS += res[3]
        elif res[2] == N:
            r+=1
            timeR += res[3]
    # Durchschnittsbildung der Absorptionszeiten
    if p != 0:
        timeP = timeP/p
    else: timeP = 'nan'
    if s != 0:
        timeS = timeS/s
    else: timeS = 'nan'
    if r != 0:
        timeR = timeR/r
    else: timeR = 'nan'
    return (p/t,s/t,r/t,timeP,timeS,timeR)

#%% Aproximiere Absorptionswahrscheinlichkeiten und -zeiten für jeden Zustand durch Simulation
pP = []
pS = []
pR = []
tP = []
tS = []
tR = []
for i in range(N+1):
    for j in range(i+1):
        res = simulateSpot(N,[j,N-i],t)
        pP.append(res[0])
        pS.append(res[1])
        pR.append(res[2])
        tP.append(res[3])
        tS.append(res[4])
        tR.append(res[5])

#%% Extrahiert die Zustände mit f_1 = N-x
def pList(x):
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

#%% Approx. Absorptionswahrscheinlichkeiten von Papier

# Umordnen
pC = np.zeros(len(pP))
pC[0] = pP[0]
pC[1] = pP[(N+1)*N//2]
pC[2] = pP[-1]
for i in range(1,N*(N+1)//2):
    pC[i+2] = pP[i]
for i in range(N*(N+1)//2+1,len(pP)-1):
    pC[i+1] = pP[i]

# Plotten
ER = np.zeros((N+1,2*N+1))
for i in range(N+1):
    for j in range(2*N+1):
        ER[i,j] = 'nan'
for i in range(N+1):
    for j in range(0,i+1):
        ER[i,N-i+2*j] = pList(i)[j]
        if j != 0:
            ER[i,N-i+2*j-1] = ER[i,N-i+2*j]

sb.heatmap(ER, cmap = "Spectral", yticklabels = False, xticklabels = False)
plt.text(-0.07*N,1.05*N,"R")
plt.text(N, -0.03*N, "S")
plt.text(2.07*N,1.05*N,"P")
plt.show()

#%% Approx. Absorptionswahrscheinlichkeiten von Schere (wie oben)

pC = np.zeros(len(pS))
pC[0] = pS[0]
pC[1] = pS[(N+1)*N//2]
pC[2] = pS[-1]
for i in range(1,N*(N+1)//2):
    pC[i+2] = pS[i]
for i in range(N*(N+1)//2+1,len(pS)-1):
    pC[i+1] = pS[i]

ER = np.zeros((N+1,2*N+1))
for i in range(N+1):
    for j in range(2*N+1):
        ER[i,j] = 'nan'
for i in range(N+1):
    for j in range(0,i+1):
        ER[i,N-i+2*j] = pList(i)[j]
        if j != 0:
            ER[i,N-i+2*j-1] = ER[i,N-i+2*j]

sb.heatmap(ER, cmap = "Spectral", yticklabels = False, xticklabels = False)
plt.text(-0.07*N,1.05*N,"R")
plt.text(N, -0.03*N, "S")
plt.text(2.07*N,1.05*N,"P")
plt.show()

#%% Approx. Absorptionswahrscheinlichkeiten von Stein (wie oben)

pC = np.zeros(len(pR))
pC[0] = pR[0]
pC[1] = pR[(N+1)*N//2]
pC[2] = pR[-1]
for i in range(1,N*(N+1)//2):
    pC[i+2] = pR[i]
for i in range(N*(N+1)//2+1,len(pS)-1):
    pC[i+1] = pR[i]

ER = np.zeros((N+1,2*N+1))
for i in range(N+1):
    for j in range(2*N+1):
        ER[i,j] = 'nan'
for i in range(N+1):
    for j in range(0,i+1):
        ER[i,N-i+2*j] = pList(i)[j]
        if j != 0:
            ER[i,N-i+2*j-1] = ER[i,N-i+2*j]

sb.heatmap(ER, cmap = "Spectral", yticklabels = False, xticklabels = False)
plt.text(-0.07*N,1.05*N,"R")
plt.text(N, -0.03*N, "S")
plt.text(2.07*N,1.05*N,"P")
plt.show()

#%% Approx. Absorptionszeiten von Papier (wie oben)

pC = np.zeros(len(tP))
pC[0] = tP[0]
pC[1] = tP[(N+1)*N//2]
pC[2] = tP[-1]
for i in range(1,N*(N+1)//2):
    pC[i+2] = tP[i]
for i in range(N*(N+1)//2+1,len(pS)-1):
    pC[i+1] = tP[i]

ER = np.zeros((N+1,2*N+1))
for i in range(N+1):
    for j in range(2*N+1):
        ER[i,j] = 'nan'
for i in range(N+1):
    for j in range(0,i+1):
        ER[i,N-i+2*j] = pList(i)[j]
        if j != 0:
            ER[i,N-i+2*j-1] = ER[i,N-i+2*j]

sb.heatmap(ER, cmap = "Spectral", yticklabels = False, xticklabels = False)
plt.text(-0.07*N,1.05*N,"R")
plt.text(N, -0.03*N, "S")
plt.text(2.07*N,1.05*N,"P")
plt.show()

#%% Approx. Absorptionszeiten von Schere (wie oben)

pC = np.zeros(len(tS))
pC[0] = tS[0]
pC[1] = tS[(N+1)*N//2]
pC[2] = tS[-1]
for i in range(1,N*(N+1)//2):
    pC[i+2] = tS[i]
for i in range(N*(N+1)//2+1,len(pS)-1):
    pC[i+1] = tS[i]

ER = np.zeros((N+1,2*N+1))
for i in range(N+1):
    for j in range(2*N+1):
        ER[i,j] = 'nan'
for i in range(N+1):
    for j in range(0,i+1):
        ER[i,N-i+2*j] = pList(i)[j]
        if j != 0:
            ER[i,N-i+2*j-1] = ER[i,N-i+2*j]

sb.heatmap(ER, cmap = "Spectral", yticklabels = False, xticklabels = False)
plt.text(-0.07*N,1.05*N,"R")
plt.text(N, -0.03*N, "S")
plt.text(2.07*N,1.05*N,"P")
plt.show()

#%% Approx. Absorptionszeiten von Stein (wie oben)

pC = np.zeros(len(tR))
pC[0] = tR[0]
pC[1] = tR[(N+1)*N//2]
pC[2] = tR[-1]
for i in range(1,N*(N+1)//2):
    pC[i+2] = tR[i]
for i in range(N*(N+1)//2+1,len(pS)-1):
    pC[i+1] = tR[i]

ER = np.zeros((N+1,2*N+1))
for i in range(N+1):
    for j in range(2*N+1):
        ER[i,j] = 'nan'
for i in range(N+1):
    for j in range(0,i+1):
        ER[i,N-i+2*j] = pList(i)[j]
        if j != 0:
            ER[i,N-i+2*j-1] = ER[i,N-i+2*j]

sb.heatmap(ER, cmap = "Spectral", yticklabels = False, xticklabels = False)
plt.text(-0.07*N,1.05*N,"R")
plt.text(N, -0.03*N, "S")
plt.text(2.07*N,1.05*N,"P")
plt.show()