"""
Script pour lancer les expériences et analyser les résultats.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import List, Dict, Callable
from datetime import datetime

from src.utils import read_tsplib_file, compute_distance_matrix
from src.models import TSPTour
from src.algorithms import (
    hill_climbing_best_improvement,
    hill_climbing_first_improvement,
    multi_start_hill_climbing,
    simulated_annealing,
    tabu_search,
    grasp,
    MetaheuristicResult
)


class Experiment:
    """Classe pour gérer les expériences."""
    
    def __init__(self, instance_path: str):
        """
        Initialise une expérience avec une instance TSP.
        
        Args:
            instance_path: Chemin vers le fichier .tsp
        """
        self.instance_path = instance_path
        self.name, self.dimension, self.coords = read_tsplib_file(instance_path)
        self.distance_matrix = compute_distance_matrix(self.coords)
        print(f"Instance chargée : {self.name} ({self.dimension} villes)")
    
    def run_algorithm(self, algorithm_name: str, algorithm_func: Callable, 
                     num_runs: int = 30, **kwargs) -> Dict:
        """
        Exécute un algorithme plusieurs fois et collecte les statistiques.
        
        Args:
            algorithm_name: Nom de l'algorithme
            algorithm_func: Fonction de l'algorithme à exécuter
            num_runs: Nombre d'exécutions indépendantes
            **kwargs: Paramètres à passer à l'algorithme
        
        Returns:
            Dictionnaire contenant les statistiques
        """
        print(f"\nExécution de {algorithm_name} ({num_runs} runs)...")
        
        results = []
        costs = []
        times = []
        evaluations_list = []
        
        for run in range(num_runs):
            # Génère un tour initial aléatoire pour chaque run
            initial_tour = TSPTour(self.distance_matrix)
            
            # Exécute l'algorithme
            result = algorithm_func(initial_tour, **kwargs)
            
            results.append(result)
            costs.append(result.best_cost)
            times.append(result.execution_time)
            evaluations_list.append(result.evaluations)
            
            if (run + 1) % 10 == 0:
                print(f"  Run {run + 1}/{num_runs} terminé")
        
        # Calcule les statistiques
        stats = {
            'algorithm': algorithm_name,
            'instance': self.name,
            'dimension': self.dimension,
            'num_runs': num_runs,
            'best_cost': int(np.min(costs)),
            'worst_cost': int(np.max(costs)),
            'mean_cost': float(np.mean(costs)),
            'std_cost': float(np.std(costs)),
            'median_cost': float(np.median(costs)),
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'mean_evaluations': float(np.mean(evaluations_list)),
            'costs': costs,
            'times': times,
            'best_tour': results[np.argmin(costs)].best_tour.tour,
            'convergence_histories': [r.convergence_history for r in results]
        }
        
        print(f"  Meilleur : {stats['best_cost']}")
        print(f"  Moyen : {stats['mean_cost']:.2f} ± {stats['std_cost']:.2f}")
        print(f"  Temps moyen : {stats['mean_time']:.4f}s")
        
        return stats
    
    def run_all_experiments(self, num_runs: int = 30, 
                          max_evaluations: int = 10000) -> List[Dict]:
        """
        Exécute tous les algorithmes sur l'instance.
        
        Args:
            num_runs: Nombre d'exécutions par algorithme
            max_evaluations: Budget d'évaluations
        
        Returns:
            Liste des statistiques pour chaque algorithme
        """
        all_stats = []
        
        # Hill-Climbing Best Improvement
        stats = self.run_algorithm(
            "Hill-Climbing (Best Improvement)",
            hill_climbing_best_improvement,
            num_runs,
            max_evaluations=max_evaluations,
            neighborhood='swap'
        )
        all_stats.append(stats)
        
        # Hill-Climbing First Improvement
        stats = self.run_algorithm(
            "Hill-Climbing (First Improvement)",
            hill_climbing_first_improvement,
            num_runs,
            max_evaluations=max_evaluations,
            neighborhood='swap'
        )
        all_stats.append(stats)
        
        # Multi-Start Hill-Climbing
        num_starts = max(5, max_evaluations // 2000)
        stats_multistart = self.run_multistart(
            num_runs, 
            num_starts=num_starts, 
            max_evaluations_per_start=max_evaluations // num_starts
        )
        all_stats.append(stats_multistart)
        
        # Simulated Annealing
        stats = self.run_algorithm(
            "Recuit Simulé",
            simulated_annealing,
            num_runs,
            max_evaluations=max_evaluations,
            initial_temp=100.0,
            alpha=0.95,
            min_temp=0.01,
            neighborhood='swap'
        )
        all_stats.append(stats)
        
        # Tabu Search
        stats = self.run_algorithm(
            "Recherche Tabou",
            tabu_search,
            num_runs,
            max_evaluations=max_evaluations,
            tabu_tenure=min(20, self.dimension // 3),
            neighborhood='swap'
        )
        all_stats.append(stats)
        
        # GRASP
        stats_grasp = self.run_grasp(
            num_runs, 
            max_iterations=max(10, max_evaluations // 1000),
            max_evaluations_local_search=max_evaluations // 10
        )
        all_stats.append(stats_grasp)
        
        return all_stats
    
    def run_multistart(self, num_runs: int, num_starts: int, 
                      max_evaluations_per_start: int) -> Dict:
        """Exécute multi-start hill-climbing."""
        print(f"\nExécution de Multi-Start Hill-Climbing ({num_runs} runs)...")
        
        results = []
        costs = []
        times = []
        evaluations_list = []
        
        for run in range(num_runs):
            result = multi_start_hill_climbing(
                self.distance_matrix,
                num_starts=num_starts,
                hc_variant='first',
                max_evaluations_per_start=max_evaluations_per_start,
                neighborhood='swap'
            )
            
            results.append(result)
            costs.append(result.best_cost)
            times.append(result.execution_time)
            evaluations_list.append(result.evaluations)
            
            if (run + 1) % 10 == 0:
                print(f"  Run {run + 1}/{num_runs} terminé")
        
        stats = {
            'algorithm': f'Multi-Start HC ({num_starts} starts)',
            'instance': self.name,
            'dimension': self.dimension,
            'num_runs': num_runs,
            'best_cost': int(np.min(costs)),
            'worst_cost': int(np.max(costs)),
            'mean_cost': float(np.mean(costs)),
            'std_cost': float(np.std(costs)),
            'median_cost': float(np.median(costs)),
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'mean_evaluations': float(np.mean(evaluations_list)),
            'costs': costs,
            'times': times,
            'best_tour': results[np.argmin(costs)].best_tour.tour,
            'convergence_histories': [r.convergence_history for r in results]
        }
        
        print(f"  Meilleur : {stats['best_cost']}")
        print(f"  Moyen : {stats['mean_cost']:.2f} ± {stats['std_cost']:.2f}")
        print(f"  Temps moyen : {stats['mean_time']:.4f}s")
        
        return stats
    
    def run_grasp(self, num_runs: int, max_iterations: int, 
                 max_evaluations_local_search: int) -> Dict:
        """Exécute GRASP."""
        print(f"\nExécution de GRASP ({num_runs} runs)...")
        
        results = []
        costs = []
        times = []
        evaluations_list = []
        
        for run in range(num_runs):
            result = grasp(
                self.distance_matrix,
                max_iterations=max_iterations,
                alpha=0.2,
                max_evaluations_local_search=max_evaluations_local_search,
                neighborhood='swap'
            )
            
            results.append(result)
            costs.append(result.best_cost)
            times.append(result.execution_time)
            evaluations_list.append(result.evaluations)
            
            if (run + 1) % 10 == 0:
                print(f"  Run {run + 1}/{num_runs} terminé")
        
        stats = {
            'algorithm': f'GRASP ({max_iterations} iterations)',
            'instance': self.name,
            'dimension': self.dimension,
            'num_runs': num_runs,
            'best_cost': int(np.min(costs)),
            'worst_cost': int(np.max(costs)),
            'mean_cost': float(np.mean(costs)),
            'std_cost': float(np.std(costs)),
            'median_cost': float(np.median(costs)),
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'mean_evaluations': float(np.mean(evaluations_list)),
            'costs': costs,
            'times': times,
            'best_tour': results[np.argmin(costs)].best_tour.tour,
            'convergence_histories': [r.convergence_history for r in results]
        }
        
        print(f"  Meilleur : {stats['best_cost']}")
        print(f"  Moyen : {stats['mean_cost']:.2f} ± {stats['std_cost']:.2f}")
        print(f"  Temps moyen : {stats['mean_time']:.4f}s")
        
        return stats
    
    def save_results(self, stats_list: List[Dict], output_dir: str = "results"):
        """
        Sauvegarde les résultats dans un fichier JSON.
        
        Args:
            stats_list: Liste des statistiques
            output_dir: Répertoire de sortie
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/{self.name}_{timestamp}.json"
        
        # Convertir les types numpy en types Python natifs
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_native(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj
        
        stats_list_native = convert_to_native(stats_list)
        
        with open(filename, 'w') as f:
            json.dump(stats_list_native, f, indent=2)
        
        print(f"\nRésultats sauvegardés dans : {filename}")
    
    def plot_results(self, stats_list: List[Dict], output_dir: str = "results"):
        """
        Génère des graphiques de comparaison.
        
        Args:
            stats_list: Liste des statistiques
            output_dir: Répertoire de sortie
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Graphique 1 : Boîtes à moustaches des coûts
        fig, ax = plt.subplots(figsize=(12, 6))
        
        algorithms = [s['algorithm'] for s in stats_list]
        costs_data = [s['costs'] for s in stats_list]
        
        bp = ax.boxplot(costs_data, tick_labels=algorithms, patch_artist=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax.set_ylabel('Coût du tour', fontsize=12)
        ax.set_title(f'Comparaison des algorithmes - {self.name}', fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        filename = f"{output_dir}/{self.name}_boxplot.png"
        plt.savefig(filename, dpi=300)
        print(f"Graphique sauvegardé : {filename}")
        plt.close()
        
        # Graphique 2 : Courbes de convergence (meilleure exécution de chaque algo)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for stats in stats_list:
            # Prend l'historique de la meilleure exécution
            best_run_idx = np.argmin(stats['costs'])
            history = stats['convergence_histories'][best_run_idx]
            
            if len(history) > 0:
                ax.plot(history, label=stats['algorithm'], linewidth=2)
        
        ax.set_xlabel('Itérations', fontsize=12)
        ax.set_ylabel('Meilleur coût trouvé', fontsize=12)
        ax.set_title(f'Courbes de convergence - {self.name}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"{output_dir}/{self.name}_convergence.png"
        plt.savefig(filename, dpi=300)
        print(f"Graphique sauvegardé : {filename}")
        plt.close()
        
        # Graphique 3 : Tableau de comparaison
        fig, ax = plt.subplots(figsize=(14, len(stats_list) * 0.6 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        headers = ['Algorithme', 'Meilleur', 'Moyen ± Écart-type', 'Médiane', 'Temps (s)']
        
        for stats in stats_list:
            row = [
                stats['algorithm'],
                str(stats['best_cost']),
                f"{stats['mean_cost']:.1f} ± {stats['std_cost']:.1f}",
                f"{stats['median_cost']:.1f}",
                f"{stats['mean_time']:.4f}"
            ]
            table_data.append(row)
        
        table = ax.table(cellText=table_data, colLabels=headers, 
                        cellLoc='left', loc='center',
                        colWidths=[0.3, 0.15, 0.25, 0.15, 0.15])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style du header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style des lignes
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title(f'Résultats comparatifs - {self.name}', 
                 fontsize=14, weight='bold', pad=20)
        plt.tight_layout()
        
        filename = f"{output_dir}/{self.name}_table.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Tableau sauvegardé : {filename}")
        plt.close()


def main():
    """Fonction principale pour lancer toutes les expériences."""
    
    # Chemins vers les instances
    instances = [
        "data/ulysses22.tsp",   # ~22 villes
        "data/berlin52.tsp",    # ~52 villes
        "data/pr76.tsp"         # ~76 villes
    ]
    
    # Budgets d'évaluations selon la taille
    budgets = {
        "ulysses22": 5000,
        "berlin52": 10000,
        "pr76": 15000
    }
    
    num_runs = 30
    
    print("=" * 70)
    print("EXPÉRIENCES TSP - COMPARAISON DE MÉTAHEURISTIQUES")
    print("=" * 70)
    
    for instance_path in instances:
        print(f"\n{'=' * 70}")
        print(f"Instance : {instance_path}")
        print(f"{'=' * 70}")
        
        # Crée l'expérience
        exp = Experiment(instance_path)
        
        # Détermine le budget
        instance_name = exp.name.lower()
        max_evals = budgets.get(instance_name, 10000)
        
        print(f"Budget d'évaluations : {max_evals}")
        print(f"Nombre de runs par algorithme : {num_runs}")
        
        # Lance toutes les expériences
        all_stats = exp.run_all_experiments(num_runs=num_runs, 
                                           max_evaluations=max_evals)
        
        # Sauvegarde et affiche les résultats
        exp.save_results(all_stats)
        exp.plot_results(all_stats)
    
    print("\n" + "=" * 70)
    print("TOUTES LES EXPÉRIENCES TERMINÉES !")
    print("=" * 70)


if __name__ == "__main__":
    main()
