"""
Modélisation du TSP et représentation des solutions.
"""

import numpy as np
import random
from typing import List


class TSPTour:
    """
    Représente un tour (solution) du TSP comme une permutation de villes.
    """
    
    def __init__(self, distance_matrix: np.ndarray, tour: List[int] = None):
        """
        Initialise un tour TSP.
        
        Args:
            distance_matrix: Matrice de distances entre les villes
            tour: Permutation des villes (None pour générer aléatoirement)
        """
        self.distance_matrix = distance_matrix
        self.n = len(distance_matrix)
        
        if tour is None:
            # Génère un tour aléatoire
            self.tour = list(range(self.n))
            random.shuffle(self.tour)
        else:
            self.tour = tour.copy()
        
        self._cost = None
    
    def cost(self) -> int:
        """
        Calcule le coût (longueur totale) du tour.
        
        Returns:
            Coût total du tour
        """
        if self._cost is None:
            total = 0
            for k in range(self.n):
                city1 = self.tour[k]
                city2 = self.tour[(k + 1) % self.n]
                total += self.distance_matrix[city1, city2]
            self._cost = total
        return self._cost
    
    def invalidate_cost(self):
        """Invalide le cache du coût."""
        self._cost = None
    
    def copy(self) -> 'TSPTour':
        """
        Crée une copie du tour.
        
        Returns:
            Nouvelle instance de TSPTour
        """
        return TSPTour(self.distance_matrix, self.tour)
    
    def swap(self, i: int, j: int):
        """
        Échange deux villes dans le tour (opération swap).
        
        Args:
            i: Indice de la première ville
            j: Indice de la seconde ville
        """
        self.tour[i], self.tour[j] = self.tour[j], self.tour[i]
        self.invalidate_cost()
    
    def two_opt(self, i: int, j: int):
        """
        Applique une opération 2-opt : inverse le segment entre i et j.
        
        Args:
            i: Indice de début
            j: Indice de fin
        """
        if i > j:
            i, j = j, i
        self.tour[i:j+1] = reversed(self.tour[i:j+1])
        self.invalidate_cost()
    
    def get_neighbors_swap(self) -> List['TSPTour']:
        """
        Génère tous les voisins possibles avec l'opération swap.
        
        Returns:
            Liste des tours voisins
        """
        neighbors = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                neighbor = self.copy()
                neighbor.swap(i, j)
                neighbors.append(neighbor)
        return neighbors
    
    def get_neighbors_2opt(self) -> List['TSPTour']:
        """
        Génère tous les voisins possibles avec l'opération 2-opt.
        
        Returns:
            Liste des tours voisins
        """
        neighbors = []
        for i in range(self.n - 1):
            for j in range(i + 1, self.n):
                neighbor = self.copy()
                neighbor.two_opt(i, j)
                neighbors.append(neighbor)
        return neighbors
    
    def __repr__(self) -> str:
        return f"TSPTour(cost={self.cost()}, tour={self.tour})"
