import json
import os
from src.experiment import (
    Experiment,
)  # Assurez-vous que le nom du fichier original est main.py


def generate_plots_only(json_path, instance_path):
    """Charge les résultats JSON et génère les graphiques sans calculs."""

    # 1. Initialiser l'objet expérience (nécessaire pour charger les coordonnées TSP)
    if not os.path.exists(instance_path):
        print(f"Erreur : Le fichier d'instance {instance_path} est introuvable.")
        return

    exp = Experiment(instance_path)

    # 2. Charger les données depuis le JSON
    if not os.path.exists(json_path):
        print(f"Erreur : Le fichier JSON {json_path} est introuvable.")
        return

    print(f"Chargement des données depuis {json_path}...")
    with open(json_path, "r") as f:
        stats_list = json.load(f)

    # 3. Générer les graphiques
    print("Génération des graphiques en cours...")
    exp.plot_results(stats_list)
    print("Terminé !")


if __name__ == "__main__":
    results_dir = "results"
    for filename in os.listdir(results_dir):
        if filename.endswith(".json"):
            # On devine le nom de l'instance à partir du nom du fichier
            # (Ex: ulysses22_2024... -> ulysses22)
            instance_name = filename.split("_")[0]
            instance_path = f"data/{instance_name}.tsp"

            generate_plots_only(os.path.join(results_dir, filename), instance_path)
