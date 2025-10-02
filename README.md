🐝 Miel & Abeilles — Optimisation par Algorithmes Génétiques

« Rien en biologie n’a de sens sauf à la lumière de l’évolution. » — Theodosius Dobzhansky

🌍 Contexte & Problématique

Une colonie d’abeilles s’installe dans un pommier sauvage, au cœur d’un champ rempli de pissenlits et de sauges des prés. Leur survie dépend de leur capacité à parcourir efficacement ce champ pour butiner et nourrir la ruche.

La ruche est située au point fixe (500, 500).

Les abeilles doivent visiter toutes les fleurs une fois et revenir à la ruche.

Le défi : minimiser la distance totale parcourue (équivalent au temps).

C’est un problème classique connu sous le nom de Voyageur de commerce (TSP).

🎯 Objectifs du projet

Implémenter une sélection naturelle numérique via un algorithme génétique (AG).

Permettre à la colonie de s’améliorer génération après génération.

Comparer différents paramétrages (mutation fixe/adaptative, sélection, crossover, élitisme).

Produire des visualisations accessibles et pédagogiques :

chemin de la meilleure abeille,

évolution des performances,

arbre généalogique.

Fournir un dépôt GitHub structuré et une présentation claire pour un public technique comme non technique.

🧩 Modélisation du problème Représentation des individus

Une abeille = un ordre de visite des fleurs (permutation des indices).

Exemple : [3, 7, 0, 1, 4] → ruche → fleur 3 → fleur 7 → … → ruche.

Fonction de fitness

Fitness = distance totale du parcours :

ruche → fleurs[ordre] → ruche

Objectif : minimiser cette distance (plus elle est petite, plus l’abeille est "fit").

Contraintes

Chaque fleur doit être visitée exactement une fois.

Le trajet doit commencer et finir à la ruche.

🧬 Algorithme génétique (AG)

Un AG est une heuristique inspirée de l’évolution naturelle :

Population initiale : génération d’abeilles avec des chemins aléatoires.

Évaluation : calcul de la fitness (distance totale).

Sélection : choix des parents (tournoi ou roulette).

Croisement : recombinaison des ordres (OX ou PMX).

Mutation : petites variations (inversion ou swap).

Élitisme : conservation des meilleurs individus.

Répétition : itération sur plusieurs générations.

⚙️ Paramètres clés

Taille de population : 100 (conformément à l’énoncé).

Sélection :

Tournoi : on choisit k=3 au hasard, le meilleur gagne.

Roulette : proba proportionnelle à 1/fitness.

Croisement (crossover) :

OX (Ordered Crossover) : copie un segment et complète dans l’ordre.

PMX (Partially Mapped Crossover) : mapping entre segments.

Mutation :

Inversion : inverse un sous-segment.

Swap : échange deux fleurs.

Taux de mutation :

Fixe (ex. 3%)

Adaptatif : augmente si stagnation trop longue.

Élitisme : ~3% des meilleurs conservés.

📊 Visualisations générées

Chemin de la meilleure abeille

Convergence de l’AG (distance moyenne et meilleure par génération)

Arbre généalogique du meilleur individu

📂 Structure du dépôt miel-abeilles/ ├─ beehive.py # Classes Field, Bee, BeehiveGA (moteur AG) ├─ main.py # Simulation + visualisations ├─ flowers.csv # Coordonnées des fleurs (extrait du Google Sheet) ├─ results/ # Résultats expérimentaux │ └─ runs/run-YYYYMMDD-HHMMSS/ │ ├─ best_path.png │ ├─ convergence.png │ ├─ genealogy.png │ ├─ metrics.csv │ └─ params.json ├─ README.md # Documentation complète └─ slides/ # Présentation (PowerPoint/Keynote/PDF)

🚀 Installation & Exécution Prérequis

Python 3.9+

Packages : numpy, pandas, matplotlib

Installation git clone https://github.com/username/miel-abeilles.git cd miel-abeilles pip install -r requirements.txt

Exécution simple python main.py --flowers flowers.csv --generations 600 --pop-size 100
--selection tournament --tournament-k 3
--crossover OX --mutation-rate 0.03 --mutation-op inversion
--elitism-rate 0.03 --adaptive-mutation --seed 42

Les résultats seront sauvegardés dans results/runs/run-YYYYMMDD-HHMMSS/.

🔬 Expérimentations à réaliser

Comparer les effets de différents paramétrages :

Mutation : 1% vs 3% vs 5% vs adaptatif

Sélection : tournoi vs roulette

Crossover : OX vs PMX

Élitisme : 1% vs 3% vs 5%

Population : 50 vs 100 vs 150

Mesures à observer :

meilleure distance atteinte

distance moyenne

vitesse de convergence

variance (stabilité)

📚 Vulgarisation (pour non spécialistes)

Un algorithme génétique, c’est comme une colonie d’abeilles :

Les meilleures butineuses montrent la voie (sélection).

On mélange leur expérience pour produire des descendants (croisement).

On garde toujours une part d’imprévu (mutation).

On s’assure de ne jamais perdre les plus fortes (élitisme).

👉 Génération après génération, la colonie devient plus rapide.

🔮 Perspectives & Améliorations

Ajouter une heuristique locale (2-opt/3-opt) pour affiner les trajets.

Explorer d’autres heuristiques :

Recuit simulé (SA)

Recherche tabou

Colonies de fourmis (ACO)

Optimisation par essaim particulaire (PSO)

Lancer en parallèle plusieurs colonies (multi-AG).

Intégrer une interface graphique interactive.

👥 Auteurs

Projet réalisé dans le cadre du Bachelor Intelligence Artificielle — La Plateforme (Marseille).

Romain Jazzar et son équipe 🐝