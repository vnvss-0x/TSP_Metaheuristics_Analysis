"""
Fonctions utilitaires pour le TSP.
"""

import numpy as np
from typing import List, Tuple


def euclidean_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Calcule la distance euclidienne entre deux points.
    
    Args:
        coord1: Coordonnées du premier point (x, y)
        coord2: Coordonnées du second point (x, y)
    
    Returns:
        Distance euclidienne arrondie à l'entier le plus proche
    """
    x1, y1 = coord1
    x2, y2 = coord2
    return round(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))


def read_tsplib_file(filepath: str) -> Tuple[str, int, List[Tuple[float, float]]]:
    """
    Lit un fichier TSPLIB et extrait les coordonnées des villes.
    
    Args:
        filepath: Chemin vers le fichier .tsp
    
    Returns:
        Tuple contenant (nom, dimension, liste des coordonnées)
    """
    name = ""
    dimension = 0
    coords = []
    
    with open(filepath, 'r') as f:
        reading_coords = False
        
        for line in f:
            line = line.strip()
            
            if line.startswith("NAME"):
                name = line.split(":")[1].strip()
            elif line.startswith("DIMENSION"):
                dimension = int(line.split(":")[1].strip())
            elif line.startswith("NODE_COORD_SECTION"):
                reading_coords = True
                continue
            elif line.startswith("EOF") or line == "":
                break
            elif reading_coords:
                parts = line.split()
                if len(parts) >= 3:
                    # Format: index x y
                    x, y = float(parts[1]), float(parts[2])
                    coords.append((x, y))
    
    return name, dimension, coords


def compute_distance_matrix(coords: List[Tuple[float, float]]) -> np.ndarray:
    """
    Calcule la matrice de distances à partir des coordonnées.
    
    Args:
        coords: Liste des coordonnées des villes
    
    Returns:
        Matrice de distances numpy array
    """
    n = len(coords)
    distance_matrix = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidean_distance(coords[i], coords[j])
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    return distance_matrix
