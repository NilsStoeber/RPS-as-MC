# RPS-as-MC

Der Code in diesem Repository ist die Implementierung meiner Bachelorarbeit.

# Modell
Wir nutzen zwei verschiedene Modelle für das evolutionäre Schere-Stein-Papier-Spiel.
In Modell 1 können Strategien aussterben, somit läuft das Spiel bis zum Verbleib einer einzigen Strategie.
In Modell 2 ist das Aussterben von Strategie nicht möglich, stattdessen gibt es immer eine positive Wahrscheinlichkeit,
dass Strategien zurückkehren können.

In beiden Modellen sind wir am Langzeitverhalten interessiert, welches in Modell 1 die Fixierung auf eine
einzige Strategie bedeutet, während wir in Modell 2 die stationäre Verteilung im Langzeitverhalten betrachten.

# Implementierung
RSPSim1.py und RSPSim2.py dienen als approximative Simulationen für diese beiden Modelle, um die gesuchten Größen anzunähern.
RSPAnalytic1.py nutzt die Fundamentalmatrix-Methode zur Berechnung von Absorptionswahrscheinlichkeiten und -zeiten in Modell 1.
RSPAnalytic2.py berechnet die nach dem Markov-Theorem eindeutige stationäre Verteilung für Modell 2.

# Weiterführend
RSPVectors.py soll die Dynamik der beiden Modelle veranschaulichen, indem mithilfe der
Übergangswahrscheinlichkeiten ein Vektorfeld erzeugt wird.
