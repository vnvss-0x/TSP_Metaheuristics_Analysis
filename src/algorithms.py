"""
Implémentation des métaheuristiques pour le TSP.
"""

import numpy as np
import random
import time
from typing import Tuple, List, Dict, Callable
from src.models import TSPTour


class MetaheuristicResult:
    """Classe pour stocker les résultats d'une exécution."""
    
    def __init__(self, best_tour: TSPTour, best_cost: int, 
                 execution_time: float, convergence_history: List[int] = None):
        self.best_tour = best_tour
        self.best_cost = best_cost
        self.execution_time = execution_time
        self.convergence_history = convergence_history or []
        self.evaluations = 0


def hill_climbing_best_improvement(initial_tour: TSPTour, 
                                   max_evaluations: int = 10000,
                                   neighborhood: str = 'swap') -> MetaheuristicResult:
    """
    Hill-climbing avec stratégie best improvement.
    
    Args:
        initial_tour: Tour de départ
        max_evaluations: Budget maximal d'évaluations
        neighborhood: Type de voisinage ('swap' ou '2opt')
    
    Returns:
        Résultat de l'optimisation
    """
    start_time = time.time()
    current_tour = initial_tour.copy()
    convergence_history = [current_tour.cost()]
    evaluations = 0
    
    while evaluations < max_evaluations:
        # Génère tous les voisins
        if neighborhood == 'swap':
            neighbors = current_tour.get_neighbors_swap()
        else:
            neighbors = current_tour.get_neighbors_2opt()
        
        # Trouve le meilleur voisin
        best_neighbor = None
        best_cost = current_tour.cost()
        
        for neighbor in neighbors:
            evaluations += 1
            neighbor_cost = neighbor.cost()
            if neighbor_cost < best_cost:
                best_cost = neighbor_cost
                best_neighbor = neighbor
            
            if evaluations >= max_evaluations:
                break
        
        # Si aucune amélioration, on arrête
        if best_neighbor is None:
            break
        
        current_tour = best_neighbor
        convergence_history.append(best_cost)
    
    execution_time = time.time() - start_time
    result = MetaheuristicResult(current_tour, current_tour.cost(), 
                                 execution_time, convergence_history)
    result.evaluations = evaluations
    return result


def hill_climbing_first_improvement(initial_tour: TSPTour, 
                                    max_evaluations: int = 10000,
                                    neighborhood: str = 'swap') -> MetaheuristicResult:
    """
    Hill-climbing avec stratégie first improvement.
    
    Args:
        initial_tour: Tour de départ
        max_evaluations: Budget maximal d'évaluations
        neighborhood: Type de voisinage ('swap' ou '2opt')
    
    Returns:
        Résultat de l'optimisation
    """
    start_time = time.time()
    current_tour = initial_tour.copy()
    current_cost = current_tour.cost()
    convergence_history = [current_cost]
    evaluations = 0
    
    while evaluations < max_evaluations:
        improved = False
        
        # Parcourt les voisins dans un ordre aléatoire
        n = current_tour.n
        if neighborhood == 'swap':
            indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
        else:
            indices = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]
        
        random.shuffle(indices)
        
        for i, j in indices:
            neighbor = current_tour.copy()
            if neighborhood == 'swap':
                neighbor.swap(i, j)
            else:
                neighbor.two_opt(i, j)
            
            evaluations += 1
            neighbor_cost = neighbor.cost()
            
            if neighbor_cost < current_cost:
                current_tour = neighbor
                current_cost = neighbor_cost
                convergence_history.append(current_cost)
                improved = True
                break
            
            if evaluations >= max_evaluations:
                break
        
        if not improved or evaluations >= max_evaluations:
            break
    
    execution_time = time.time() - start_time
    result = MetaheuristicResult(current_tour, current_cost, 
                                 execution_time, convergence_history)
    result.evaluations = evaluations
    return result


def multi_start_hill_climbing(distance_matrix: np.ndarray, 
                              num_starts: int = 10,
                              hc_variant: str = 'best',
                              max_evaluations_per_start: int = 1000,
                              neighborhood: str = 'swap') -> MetaheuristicResult:
    """
    Multi-start hill-climbing.
    
    Args:
        distance_matrix: Matrice de distances
        num_starts: Nombre de départs aléatoires
        hc_variant: Variante de HC ('best' ou 'first')
        max_evaluations_per_start: Budget par départ
        neighborhood: Type de voisinage ('swap' ou '2opt')
    
    Returns:
        Résultat de l'optimisation
    """
    start_time = time.time()
    best_overall_tour = None
    best_overall_cost = float('inf')
    convergence_history = []
    total_evaluations = 0
    
    hc_function = hill_climbing_best_improvement if hc_variant == 'best' else hill_climbing_first_improvement
    
    for i in range(num_starts):
        # Génère un tour aléatoire
        initial_tour = TSPTour(distance_matrix)
        
        # Lance hill-climbing
        result = hc_function(initial_tour, max_evaluations_per_start, neighborhood)
        total_evaluations += result.evaluations
        
        if result.best_cost < best_overall_cost:
            best_overall_cost = result.best_cost
            best_overall_tour = result.best_tour
        
        convergence_history.extend(result.convergence_history)
    
    execution_time = time.time() - start_time
    result = MetaheuristicResult(best_overall_tour, best_overall_cost, 
                                 execution_time, convergence_history)
    result.evaluations = total_evaluations
    return result


def simulated_annealing(initial_tour: TSPTour, 
                       max_evaluations: int = 10000,
                       initial_temp: float = 100.0,
                       alpha: float = 0.95,
                       min_temp: float = 0.01,
                       neighborhood: str = 'swap') -> MetaheuristicResult:
    """
    Recuit simulé pour le TSP.
    
    Args:
        initial_tour: Tour de départ
        max_evaluations: Budget maximal d'évaluations
        initial_temp: Température initiale (T0)
        alpha: Coefficient de refroidissement
        min_temp: Température minimale
        neighborhood: Type de voisinage ('swap' ou '2opt')
    
    Returns:
        Résultat de l'optimisation
    """
    start_time = time.time()
    current_tour = initial_tour.copy()
    current_cost = current_tour.cost()
    
    best_tour = current_tour.copy()
    best_cost = current_cost
    
    temperature = initial_temp
    convergence_history = [best_cost]
    evaluations = 0
    
    while evaluations < max_evaluations and temperature > min_temp:
        # Génère un voisin aléatoire
        neighbor = current_tour.copy()
        n = neighbor.n
        
        if neighborhood == 'swap':
            i, j = random.sample(range(n), 2)
            neighbor.swap(i, j)
        else:
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
            neighbor.two_opt(i, j)
        
        evaluations += 1
        neighbor_cost = neighbor.cost()
        delta = neighbor_cost - current_cost
        
        # Acceptation ou rejet
        if delta <= 0 or random.random() < np.exp(-delta / temperature):
            current_tour = neighbor
            current_cost = neighbor_cost
            
            if current_cost < best_cost:
                best_tour = current_tour.copy()
                best_cost = current_cost
                convergence_history.append(best_cost)
        
        # Refroidissement
        temperature *= alpha
    
    execution_time = time.time() - start_time
    result = MetaheuristicResult(best_tour, best_cost, 
                                 execution_time, convergence_history)
    result.evaluations = evaluations
    return result


def tabu_search(initial_tour: TSPTour, 
               max_evaluations: int = 10000,
               tabu_tenure: int = 10,
               neighborhood: str = 'swap') -> MetaheuristicResult:
    """
    Recherche tabou pour le TSP.
    
    Args:
        initial_tour: Tour de départ
        max_evaluations: Budget maximal d'évaluations
        tabu_tenure: Durée de tabou
        neighborhood: Type de voisinage ('swap' ou '2opt')
    
    Returns:
        Résultat de l'optimisation
    """
    start_time = time.time()
    current_tour = initial_tour.copy()
    current_cost = current_tour.cost()
    
    best_tour = current_tour.copy()
    best_cost = current_cost
    
    # Liste tabou : stocke les mouvements récents (i, j)
    tabu_list = []
    convergence_history = [best_cost]
    evaluations = 0
    
    while evaluations < max_evaluations:
        n = current_tour.n
        
        # Génère les indices de mouvements possibles
        if neighborhood == 'swap':
            moves = [(i, j) for i in range(n) for j in range(i + 1, n)]
        else:
            moves = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]
        
        best_neighbor = None
        best_neighbor_cost = float('inf')
        best_move = None
        
        # Cherche le meilleur voisin non-tabou
        for move in moves:
            i, j = move
            neighbor = current_tour.copy()
            
            if neighborhood == 'swap':
                neighbor.swap(i, j)
            else:
                neighbor.two_opt(i, j)
            
            evaluations += 1
            neighbor_cost = neighbor.cost()
            
            # Accepte si non-tabou ou si critère d'aspiration (meilleure solution globale)
            if (move not in tabu_list or neighbor_cost < best_cost):
                if neighbor_cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = neighbor_cost
                    best_move = move
            
            if evaluations >= max_evaluations:
                break
        
        if best_neighbor is None:
            break
        
        # Mise à jour
        current_tour = best_neighbor
        current_cost = best_neighbor_cost
        
        # Ajoute le mouvement à la liste tabou
        tabu_list.append(best_move)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
        
        # Mise à jour de la meilleure solution
        if current_cost < best_cost:
            best_tour = current_tour.copy()
            best_cost = current_cost
            convergence_history.append(best_cost)
    
    execution_time = time.time() - start_time
    result = MetaheuristicResult(best_tour, best_cost, 
                                 execution_time, convergence_history)
    result.evaluations = evaluations
    return result


def grasp(distance_matrix: np.ndarray, 
         max_iterations: int = 10,
         alpha: float = 0.2,
         max_evaluations_local_search: int = 1000,
         neighborhood: str = 'swap') -> MetaheuristicResult:
    """
    GRASP (Greedy Randomized Adaptive Search Procedure) pour le TSP.
    
    Args:
        distance_matrix: Matrice de distances
        max_iterations: Nombre d'itérations
        alpha: Paramètre de randomisation (0 = glouton, 1 = aléatoire)
        max_evaluations_local_search: Budget pour la recherche locale
        neighborhood: Type de voisinage ('swap' ou '2opt')
    
    Returns:
        Résultat de l'optimisation
    """
    start_time = time.time()
    best_overall_tour = None
    best_overall_cost = float('inf')
    convergence_history = []
    total_evaluations = 0
    
    n = len(distance_matrix)
    
    for iteration in range(max_iterations):
        # Phase de construction gloutonne randomisée
        unvisited = set(range(n))
        current_city = random.choice(list(unvisited))
        tour = [current_city]
        unvisited.remove(current_city)
        
        while unvisited:
            # Calcule les distances vers toutes les villes non visitées
            distances = [(city, distance_matrix[current_city, city]) 
                        for city in unvisited]
            distances.sort(key=lambda x: x[1])
            
            # Crée une RCL (Restricted Candidate List)
            min_dist = distances[0][1]
            max_dist = distances[-1][1]
            threshold = min_dist + alpha * (max_dist - min_dist)
            
            rcl = [city for city, dist in distances if dist <= threshold]
            
            # Choisit aléatoirement dans la RCL
            next_city = random.choice(rcl)
            tour.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city
        
        # Crée un tour TSP à partir de la construction
        constructed_tour = TSPTour(distance_matrix, tour)
        
        # Phase de recherche locale (hill-climbing)
        result = hill_climbing_first_improvement(
            constructed_tour, 
            max_evaluations_local_search, 
            neighborhood
        )
        total_evaluations += result.evaluations
        
        if result.best_cost < best_overall_cost:
            best_overall_cost = result.best_cost
            best_overall_tour = result.best_tour
        
        convergence_history.extend(result.convergence_history)
    
    execution_time = time.time() - start_time
    result = MetaheuristicResult(best_overall_tour, best_overall_cost, 
                                 execution_time, convergence_history)
    result.evaluations = total_evaluations
    return result
