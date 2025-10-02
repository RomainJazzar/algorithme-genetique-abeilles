
# Miel & Abeilles — Algorithme génétique sur un TSP ruche→fleurs→ruche

> « Rien en biologie n’a de sens sauf à la lumière de l’évolution. » — Dobzhansky

## Problème
Une ruche en (500,500) doit visiter **toutes** les fleurs d’un champ (coordonnées fournies) **une seule fois**, puis **revenir** à la ruche, en **minimisant** le temps de parcours (≈ distance si vitesse constante). C’est un **TSP**.

## Modèle & Représentation
- **Individu (abeille)** : permutation des indices des fleurs (ordre de visite).
- **Fitness** : distance totale ruche→fleurs(order)→ruche (à **minimiser**).

## Algorithme (Sélection naturelle numérique)
- **Sélection** : tournoi *k*=3 (ou roulette).
- **Croisement** : OX (Ordered Crossover) ou PMX (Partially Mapped Crossover).
- **Mutation** : inversion (ou swap), taux par individu (par défaut 3%).
- **Élitisme** : on conserve les ~3% meilleurs intacts.
- **Mutation adaptative (option)** : augmente si la convergence stagne.

## Fichiers
- `beehive.py` : Field, Bee, BeehiveGA (moteur AG + généalogie).
- `main.py` : chargement des données, exécution, **visualisations** :
  - `best_path.png` : points (fleurs) + chemin de la **meilleure abeille**.
  - `convergence.png` : distances *best* et *moyennes* par génération.
  - `genealogy.png` : arbre d’ascendance du meilleur individu.
- `flowers.csv` : coordonnées des fleurs (*remplacer le fichier de démo*).

## Données (ETL minimal & RGPD)
- Format : CSV avec colonnes `x,y` (et optionnellement `id`).
- `main.py` valide la présence de `x,y` et supprime les `NaN`.
- Pas de données personnelles → conformité RGPD triviale. Traçabilité assurée via `results/runs/*` (params + métriques).

## Installation & Exécution
```bash
# (Optionnel) créer un venv puis installer numpy, pandas, matplotlib
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install numpy pandas matplotlib

# Dans le dossier du projet
python main.py --flowers flowers.csv --generations 600 --pop-size 100 \
  --selection tournament --tournament-k 3 --crossover OX --mutation-rate 0.03 \
  --mutation-op inversion --elitism-rate 0.03 --adaptive-mutation --seed 42
```

Les sorties sont écrites dans `results/runs/run-YYYYMMDD-HHMMSS/` :
- `best_path.png` — chemin final
- `convergence.png` — courbe d’évolution
- `genealogy.png` — arbre généalogique
- `metrics.csv` — meilleures/moyennes par génération
- `params.json` — tous les paramètres d’exécution

## Justification des choix
- **Permutation** : garantit la visite unique de chaque fleur.
- **Fitness = distance** : correspond au “temps de parcours” demandé.
- **OX/PMX** : crossovers adaptés aux permutations (pas de duplications).
- **Inversion** : mutation efficace pour TSP, corrige les « zigzags ».
- **Tournoi** : robuste, simple, peu sensible à l’échelle des fitness.
- **Élitisme** : empêche la régression, accélère la convergence.
- **Mutation adaptative** : limite le blocage dans un optimum local.

## Comparaisons à réaliser (expérimentation)
- Mutation : 1% vs 3% vs 5% vs adaptatif.
- Sélection : tournoi (k=3) vs roulette.
- Crossover : OX vs PMX.
- Élitisme : 1% vs 3% vs 5%.
- Population : 50 vs 100 vs 150.

Comparer **meilleure distance**, **distance moyenne**, **vitesse de convergence** et **stabilité** (variance). Documenter le **paramétrage retenu**.

## Accessibilité (WCAG) & Présentation
- Titres, axes, légendes sur chaque figure.
- Police et tailles lisibles ; contrastes par défaut (Matplotlib).
- Dans les slides : alt‑text pour les images, ordre logique et synthétique.

## Pistes d’amélioration (veille)
- 2‑opt/3‑opt en post‑traitement pour affiner un tour.
- Recuit simulé, recherche tabou, ACO (colonies de fourmis), PSO.
- Hybridation AG + heuristiques locales.

---

**Note** : Remplacez `flowers.csv` par le CSV des coordonnées fourni par l’énoncé (la démo incluse est aléatoire).
