# TSP Metaheuristics Analysis

Analyse comparative de métaheuristiques pour le problème du voyageur de commerce (TSP).

## 📋 Description

Ce projet implémente et compare plusieurs métaheuristiques pour résoudre le TSP :

- **Hill-Climbing** (Best Improvement et First Improvement)
- **Multi-Start Hill-Climbing**
- **Recuit Simulé (Simulated Annealing)**
- **Recherche Tabou (Tabu Search)**
- **GRASP (Greedy Randomized Adaptive Search Procedure)**

## 🏗️ Structure du projet

```
TSP_Metaheuristics_Analysis/
├── data/                  # Instances TSP (TSPLIB format)
│   ├── ulysses22.tsp     # 22 villes
│   ├── berlin52.tsp      # 52 villes
│   └── pr76.tsp          # 76 villes
├── src/                   # Code source
│   ├── __init__.py
│   ├── utils.py          # Lecture de fichiers et calcul de distances
│   ├── models.py         # Représentation d'un tour TSP
│   ├── algorithms.py     # Implémentation des métaheuristiques
│   └── experiment.py     # Script d'expérimentation
├── results/              # Résultats (JSON, graphiques)
├── report/               # Rapport
├── requirements.txt      # Dépendances Python
└── README.md
```

## 🔧 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip

### Installation des dépendances

```bash
pip install -r requirements.txt
```

## 🚀 Utilisation

### Lancer toutes les expériences

Pour exécuter les 30 runs de chaque algorithme sur toutes les instances :

```bash
python -m src.experiment
```

Cette commande va :
1. Charger les 3 instances TSP
2. Exécuter chaque algorithme 30 fois sur chaque instance
3. Calculer les statistiques (meilleur, moyen, écart-type, temps)
4. Générer des graphiques de comparaison
5. Sauvegarder les résultats en JSON

### Résultats

Les résultats sont sauvegardés dans le dossier `results/` :
- **Fichiers JSON** : Statistiques détaillées pour chaque instance
- **Graphiques** :
  - `*_boxplot.png` : Distribution des coûts pour chaque algorithme
  - `*_convergence.png` : Courbes de convergence
  - `*_table.png` : Tableau comparatif des performances

## 📊 Métaheuristiques implémentées

### 1. Hill-Climbing (HC)

Deux variantes :
- **Best Improvement** : Explore tout le voisinage et choisit le meilleur
- **First Improvement** : S'arrête au premier voisin améliorant

**Voisinage** : Swap de deux villes

### 2. Multi-Start Hill-Climbing

Lance plusieurs Hill-Climbing depuis différentes solutions initiales aléatoires et conserve la meilleure.

### 3. Recuit Simulé (SA)

**Paramètres** :
- Température initiale : T₀ = 100
- Coefficient de refroidissement : α = 0.95
- Température minimale : Tₘᵢₙ = 0.01
- Critère d'acceptation : P(ΔE) = exp(-ΔE/T)

### 4. Recherche Tabou (TS)

**Paramètres** :
- Durée de tabou : tenure = min(20, n/3)
- Critère d'aspiration : Accepte si meilleure solution globale

### 5. GRASP

**Phases** :
1. Construction gloutonne randomisée (RCL avec α = 0.2)
2. Recherche locale (Hill-Climbing First Improvement)

## 🔬 Protocole expérimental

Pour chaque algorithme et chaque instance :
- **30 exécutions indépendantes** avec des solutions initiales différentes
- **Budget d'évaluations** :
  - ulysses22 : 5 000 évaluations
  - berlin52 : 10 000 évaluations
  - pr76 : 15 000 évaluations

**Métriques mesurées** :
- Meilleur coût obtenu
- Coût moyen et écart-type
- Médiane
- Temps d'exécution moyen
- Courbes de convergence

## 📈 Exemple de résultats

```
Instance : berlin52 (52 villes)

Algorithme                      | Meilleur | Moyen ± σ      | Temps (s)
--------------------------------|----------|----------------|----------
HC (Best Improvement)           | 8532     | 9125.3 ± 425.1 | 0.2341
HC (First Improvement)          | 8421     | 8987.6 ± 387.2 | 0.1523
Multi-Start HC (5 starts)       | 8234     | 8612.4 ± 298.5 | 0.4567
Recuit Simulé                   | 7956     | 8234.7 ± 215.3 | 0.3421
Recherche Tabou                 | 7842     | 8112.3 ± 189.4 | 0.4123
GRASP (10 iterations)           | 7923     | 8198.5 ± 201.2 | 0.3876
```

## 🧪 Personnalisation des expériences

Pour modifier les paramètres, éditez le fichier `src/experiment.py` :

```python
# Modifier les budgets d'évaluations
budgets = {
    "ulysses22": 5000,
    "berlin52": 10000,
    "pr76": 15000
}

# Modifier le nombre de runs
num_runs = 30

# Modifier les paramètres du recuit simulé
initial_temp=100.0,
alpha=0.95,
min_temp=0.01
```

## 📚 Instances TSPLIB

Les instances sont au format TSPLIB standard :
- **ulysses22** : 22 villes, solution optimale = 7013
- **berlin52** : 52 villes, solution optimale = 7542
- **pr76** : 76 villes, solution optimale = 108159

## 👥 Auteurs

Ce projet a été réalisé par :
* **Anass MAKHLOUK** — [@vnvss-0x](https://github.com/vnvss-0x)
* **Houdaifa BAHOU** — [@bahouhoudaifa](https://github.com/bahouhoudaifa)
* **Anass RWCHI** — [@ANAS-RWICHI](https://github.com/ANAS-RWICHI)