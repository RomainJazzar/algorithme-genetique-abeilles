
"""
beehive.py — Miel & Abeilles (Algorithme génétique sur un TSP "ruche → fleurs → ruche")

Ce module contient :
- Field : le champ (points des fleurs + ruche) et les calculs de distance
- Bee : un individu (chromosome = permutation d'indices de fleurs) + méta (id, gén, parents)
- BeehiveGA : l'algorithme génétique (sélection, croisement, mutation, élitisme, historique)

Conception (Pourquoi comme ça) :
- Un trajet optimal qui part de la ruche (500,500), visite chaque fleur une fois et revient à la ruche
  est un problème de voyageur de commerce (TSP). Représenter une abeille comme une permutation
  est donc naturel : l'ordre encode le chemin.
- La fitness est la distance totale (équivalente à "temps" si vitesse constante), que l'on MINIMISE.
- Les opérateurs (crossover OX/PMX, mutation inversion/swap) sont choisis car ils préservent la
  validité des permutations (pas de doublons/omissions).

Dépendances : numpy (calculs), random, math

Les fonctions de visualisation sont gérées dans main.py pour séparer "moteur" et "I/O/plots".
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Iterable
import math, random
import numpy as np

# -----------------------------
# Modèle du champ et métriques
# -----------------------------

@dataclass
class Field:
    flowers: np.ndarray  # shape (N, 2) — colonnes x, y
    hive: Tuple[float, float] = (500.0, 500.0)

    def __post_init__(self):
        assert self.flowers.ndim == 2 and self.flowers.shape[1] == 2, "flowers doit être (N,2)"
        self.flowers = self.flowers.astype(float)

    @staticmethod
    def euclid(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return math.hypot(dx, dy)

    def tour_length(self, order: Iterable[int]) -> float:
        """Distance totale ruche → fleurs(order) → ruche."""
        o = list(order)
        total = 0.0
        # ruche -> première fleur
        total += self.euclid(self.hive, tuple(self.flowers[o[0]]))
        # parcours interne
        for a, b in zip(o[:-1], o[1:]):
            total += self.euclid(tuple(self.flowers[a]), tuple(self.flowers[b]))
        # dernière fleur -> ruche
        total += self.euclid(tuple(self.flowers[o[-1]]), self.hive)
        return total


# -----------------------------
# Individu (abeille)
# -----------------------------

@dataclass
class Bee:
    order: np.ndarray                         # permutation d'indices de fleurs
    fitness: Optional[float] = None           # distance totale (à minimiser)
    id: Optional[int] = None                  # identifiant unique
    generation: Optional[int] = None          # numéro de génération
    parents: Tuple[Optional[int], Optional[int]] = (None, None)  # ids des parents

    def clone(self) -> "Bee":
        return Bee(order=self.order.copy(), fitness=self.fitness, id=self.id,
                   generation=self.generation, parents=self.parents)


# ------------------------------------
# Algorithme génétique (moteur)
# ------------------------------------

class BeehiveGA:
    def __init__(
        self,
        field: Field,
        pop_size: int = 100,
        selection: str = "tournament",  # "tournament" ou "roulette"
        tournament_k: int = 3,
        crossover: str = "OX",          # "OX" ou "PMX"
        mutation_rate: float = 0.03,
        mutation_op: str = "inversion", # "inversion" ou "swap"
        elitism_rate: float = 0.03,
        adaptive_mutation: bool = False,
        seed: Optional[int] = None,
    ):
        self.field = field
        self.pop_size = pop_size
        self.selection = selection
        self.tournament_k = tournament_k
        self.crossover = crossover
        self.mutation_rate = mutation_rate
        self.mutation_op = mutation_op
        self.elitism_rate = elitism_rate
        self.adaptive_mutation = adaptive_mutation

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self._id_counter = 0
        self.genealogy: Dict[int, Tuple[Optional[int], Optional[int], int]] = {}  # id -> (p1, p2, gen)

        # historisation
        self.history_best: List[float] = []
        self.history_avg: List[float] = []
        self.history_best_ids: List[int] = []

    # ---------- init & évaluation

    def _new_id(self) -> int:
        i = self._id_counter
        self._id_counter += 1
        return i

    def init_population(self) -> List[Bee]:
        n = len(self.field.flowers)
        pop = []
        for _ in range(self.pop_size):
            perm = np.arange(n)
            self.np_rng.shuffle(perm)
            bid = self._new_id()
            bee = Bee(order=perm, id=bid, generation=0, parents=(None, None))
            self.genealogy[bid] = (None, None, 0)
            pop.append(bee)
        return pop

    def evaluate(self, population: List[Bee]) -> None:
        for bee in population:
            bee.fitness = self.field.tour_length(bee.order)

    # ---------- sélection

    def _select_parent(self, population: List[Bee]) -> Bee:
        if self.selection == "tournament":
            subset = self.rng.sample(population, k=self.tournament_k)
            return min(subset, key=lambda b: b.fitness)
        elif self.selection == "roulette":
            # probabilités ~ 1/fitness (plus court = plus probable)
            inv = np.array([1.0 / (b.fitness + 1e-9) for b in population])
            probs = inv / inv.sum()
            idx = np.random.choice(len(population), p=probs)
            return population[idx]
        else:
            raise ValueError("selection doit être 'tournament' ou 'roulette'")

    # ---------- crossover (pour permutations)

    def _crossover(self, p1: Bee, p2: Bee) -> np.ndarray:
        if self.crossover.upper() == "OX":
            return self._ox(p1.order, p2.order)
        elif self.crossover.upper() == "PMX":
            return self._pmx(p1.order, p2.order)
        else:
            raise ValueError("crossover doit être 'OX' ou 'PMX'")

    def _ox(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = len(a)
        i, j = sorted(self.rng.sample(range(n), 2))
        child = np.full(n, -1, dtype=int)
        # copier segment de a
        child[i:j+1] = a[i:j+1]
        # compléter avec l'ordre de b
        b_seq = [x for x in b if x not in child]
        # remplir à partir de j+1 circulairement
        idxs = list(range(j+1, n)) + list(range(0, i))
        for k, pos in enumerate(idxs):
            child[pos] = b_seq[k]
        return child

    def _pmx(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        n = len(a)
        i, j = sorted(self.rng.sample(range(n), 2))
        child = np.full(n, -1, dtype=int)
        # copier segment de a
        child[i:j+1] = a[i:j+1]
        # construire mapping b->a sur le segment
        mapping = {b[t]: a[t] for t in range(i, j+1)}
        # compléter positions restantes en parcourant b
        for pos in list(range(0, i)) + list(range(j+1, n)):
            val = b[pos]
            # Résoudre les collisions via mapping
            while val in mapping and val in child:
                val = mapping[val]
            if val in child:
                # injecter la première valeur manquante (fallback sûr)
                for cand in a:
                    if cand not in child:
                        val = cand
                        break
            child[pos] = val
        return child

    # ---------- mutation

    def _mutate(self, ord_: np.ndarray) -> None:
        if self.rng.random() > self.mutation_rate:
            return
        n = len(ord_)
        i, j = sorted(self.rng.sample(range(n), 2))
        if self.mutation_op == "inversion":
            ord_[i:j+1] = ord_[i:j+1][::-1]
        elif self.mutation_op == "swap":
            ord_[i], ord_[j] = ord_[j], ord_[i]
        else:
            raise ValueError("mutation_op doit être 'inversion' ou 'swap'")

    # ---------- génération suivante

    def _elitism_count(self) -> int:
        e = int(round(self.elitism_rate * self.pop_size))
        return max(1, e)

    def next_generation(self, population: List[Bee], gen: int) -> List[Bee]:
        # trier pour élitisme
        pop_sorted = sorted(population, key=lambda b: b.fitness)
        elites = [b.clone() for b in pop_sorted[: self._elitism_count()]]
        # enfants
        children: List[Bee] = []
        while len(elites) + len(children) < self.pop_size:
            p1 = self._select_parent(population)
            p2 = self._select_parent(population)
            child_order = self._crossover(p1, p2)
            # mutation (in-place)
            self._mutate(child_order)
            cid = self._new_id()
            child = Bee(order=child_order, id=cid, generation=gen, parents=(p1.id, p2.id))
            self.genealogy[cid] = (p1.id, p2.id, gen)
            children.append(child)
        new_pop = elites + children
        self.evaluate(new_pop)
        return new_pop

    # ---------- boucle principale

    def run(self, n_generations: int = 600):
        population = self.init_population()
        self.evaluate(population)

        best_overall = min(population, key=lambda b: b.fitness)
        no_improve = 0
        base_mut_rate = self.mutation_rate

        for g in range(1, n_generations + 1):
            # stats
            best = min(population, key=lambda b: b.fitness)
            avg = float(np.mean([b.fitness for b in population]))
            self.history_best.append(best.fitness)
            self.history_avg.append(avg)
            self.history_best_ids.append(best.id)

            if best.fitness + 1e-9 < best_overall.fitness:
                best_overall = best.clone()
                no_improve = 0
                # si mutation adaptative : revenir au taux de base
                if self.adaptive_mutation:
                    self.mutation_rate = base_mut_rate
            else:
                no_improve += 1
                if self.adaptive_mutation and no_improve >= 50:
                    # booster temporaire la mutation pour échapper aux minima locaux
                    self.mutation_rate = min(0.15, self.mutation_rate * 1.5)

            # évolution
            population = self.next_generation(population, gen=g)

        # stats finales (après dernière génération produite)
        best = min(population, key=lambda b: b.fitness)
        avg = float(np.mean([b.fitness for b in population]))
        self.history_best.append(best.fitness)
        self.history_avg.append(avg)
        self.history_best_ids.append(best.id)

        return population, {
            "best_per_gen": self.history_best,
            "avg_per_gen": self.history_avg,
            "best_ids": self.history_best_ids,
            "genealogy": self.genealogy,
        }
