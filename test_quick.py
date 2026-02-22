"""
Script de test rapide pour vérifier que tout fonctionne.
"""

from src.utils import read_tsplib_file, compute_distance_matrix
from src.models import TSPTour
from src.algorithms import (
    hill_climbing_best_improvement,
    hill_climbing_first_improvement,
    simulated_annealing
)


def test_basic_functionality():
    """Test basique de fonctionnalité."""
    
    print("=" * 60)
    print("TEST RAPIDE DU SYSTÈME")
    print("=" * 60)
    
    # 1. Charger une instance
    print("\n1. Chargement de l'instance ulysses22...")
    name, dimension, coords = read_tsplib_file("data/ulysses22.tsp")
    print(f"   ✓ Instance chargée : {name} ({dimension} villes)")
    
    # 2. Calculer la matrice de distances
    print("\n2. Calcul de la matrice de distances...")
    distance_matrix = compute_distance_matrix(coords)
    print(f"   ✓ Matrice {distance_matrix.shape} créée")
    
    # 3. Créer un tour aléatoire
    print("\n3. Création d'un tour aléatoire...")
    tour = TSPTour(distance_matrix)
    print(f"   ✓ Tour créé, coût = {tour.cost()}")
    
    # 4. Tester Hill-Climbing Best
    print("\n4. Test Hill-Climbing (Best Improvement)...")
    result = hill_climbing_best_improvement(tour, max_evaluations=500)
    print(f"   ✓ Coût final = {result.best_cost}")
    print(f"   ✓ Temps = {result.execution_time:.4f}s")
    print(f"   ✓ Évaluations = {result.evaluations}")
    
    # 5. Tester Hill-Climbing First
    print("\n5. Test Hill-Climbing (First Improvement)...")
    tour2 = TSPTour(distance_matrix)
    result = hill_climbing_first_improvement(tour2, max_evaluations=500)
    print(f"   ✓ Coût final = {result.best_cost}")
    print(f"   ✓ Temps = {result.execution_time:.4f}s")
    
    # 6. Tester Recuit Simulé
    print("\n6. Test Recuit Simulé...")
    tour3 = TSPTour(distance_matrix)
    result = simulated_annealing(tour3, max_evaluations=500)
    print(f"   ✓ Coût final = {result.best_cost}")
    print(f"   ✓ Temps = {result.execution_time:.4f}s")
    
    print("\n" + "=" * 60)
    print("✓ TOUS LES TESTS SONT PASSÉS AVEC SUCCÈS !")
    print("=" * 60)
    print("\nVous pouvez maintenant lancer les expériences complètes avec :")
    print("  python -m src.experiment")
    print("=" * 60)


if __name__ == "__main__":
    test_basic_functionality()
