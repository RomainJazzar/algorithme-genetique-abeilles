"""
main.py — Simulation de l'adaptation des abeilles (AG) + visualisations
- Charge flowers.csv (x,y)
- Lance l'algorithme génétique (BeehiveGA)
- Sauvegarde 3 figures : best_path.png, convergence.png, genealogy.png
- Log des paramètres + métriques dans results/runs/<timestamp>/
"""

import argparse, os, json, datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from beehive import Field, BeehiveGA, Bee


def load_flowers(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    # ETL minimal : colonnes obligatoires x,y
    if not set(['x','y']).issubset(df.columns):
        raise ValueError("Le fichier CSV doit contenir des colonnes 'x' et 'y'.")
    df = df[['x','y']].dropna()
    arr = df.to_numpy(dtype=float)
    return arr


def plot_best_path(field: Field, bee: Bee, out_path: str):
    plt.figure()
    pts = field.flowers
    # points des fleurs
    plt.scatter(pts[:,0], pts[:,1], s=20, label="Fleurs", marker='x')
    # ruche
    plt.scatter([field.hive[0]], [field.hive[1]], marker='s', s=60, label="Ruche")
    # chemin
    order = bee.order.tolist()
    path_x = [field.hive[0]] + [pts[i,0] for i in order] + [field.hive[0]]
    path_y = [field.hive[1]] + [pts[i,1] for i in order] + [field.hive[1]]
    plt.plot(path_x, path_y, linewidth=1)
    plt.title(f"Meilleure abeille — distance = {bee.fitness:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_convergence(best_list, avg_list, out_path: str):
    gens = list(range(len(best_list)))
    plt.figure()
    plt.plot(gens, avg_list, label="Distance moyenne / génération")
    plt.plot(gens, best_list, linestyle='--', label="Meilleure distance / génération")
    plt.xlabel("Génération")
    plt.ylabel("Distance (≈ temps)")
    plt.title("Convergence de l'AG")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_genealogy(genealogy: dict, best_id: int, out_path: str, mode: str = "lineage", max_edges: int = 2000):
    """
    Genealogy plot with two modes:
      - "lineage" (default): show ONLY the direct ancestor chain of best_id.
      - "full": show a capped number of edges across all ancestors (max_edges).
    This prevents matplotlib freezes with very dense graphs on Windows.
    """
    if not genealogy or best_id is None:
        plt.figure()
        plt.title("Généalogie indisponible")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    if mode == "lineage":
        # Build the direct lineage: best -> parent -> grandparent -> ...
        chain = []
        seen = set()
        cur = best_id
        while cur is not None and cur not in seen:
            seen.add(cur)
            p1, p2, gen = genealogy.get(cur, (None, None, None))
            chain.append((cur, gen))
            # Prefer p1 if exists, else p2
            cur = p1 if p1 is not None else p2

        # Normalize gens (may be None at roots)
        gens = [g for _, g in chain if g is not None]
        gmin = min(gens) if gens else 0

        xs = list(range(len(chain)))          # simple horizontal layout
        ys = [g if g is not None else gmin for _, g in chain]

        plt.figure()
        plt.plot(xs, ys, linewidth=2)
        plt.scatter(xs, ys, s=30)
        # Mark the best node
        plt.scatter([xs[0]], [ys[0]], s=80, marker='s')
        plt.annotate("BEST", (xs[0], ys[0]), xytext=(5, 5), textcoords="offset points")

        plt.gca().invert_yaxis()  # gen 0 at bottom if present
        plt.xlabel("Étapes dans la lignée (best → ancêtres)")
        plt.ylabel("Génération")
        plt.title("Arbre généalogique — lignée directe du meilleur")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    # FULL mode (capped breadth-first)
    edges = []
    nodes = set()
    gens = {}
    queue = [best_id]
    seen = set()
    while queue and len(edges) < max_edges:
        cid = queue.pop(0)
        if cid in seen:
            continue
        seen.add(cid)
        nodes.add(cid)
        p1, p2, gen = genealogy.get(cid, (None, None, None))
        if gen is not None:
            gens[cid] = gen
        for p in (p1, p2):
            if p is not None:
                edges.append((cid, p))
                if len(edges) >= max_edges:
                    break
                if p not in seen:
                    queue.append(p)

    if not gens:
        plt.figure()
        plt.title("Généalogie (aucune info de génération)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    by_gen = {}
    for nid, g in gens.items():
        by_gen.setdefault(g, []).append(nid)
    for g in by_gen:
        by_gen[g].sort()

    coords = {}
    for g in sorted(by_gen.keys()):
        ids = by_gen[g]
        for k, nid in enumerate(ids):
            coords[nid] = (k, g)

    plt.figure()
    # edges
    for child, parent in edges:
        if child in coords and parent in coords:
            x1, y1 = coords[child]
            x2, y2 = coords[parent]
            plt.plot([x1, x2], [y1, y2], linewidth=0.8)
    # nodes
    xs = [coords[n][0] for n in coords]
    ys = [coords[n][1] for n in coords]
    plt.scatter(xs, ys, s=12)
    if best_id in coords:
        bx, by = coords[best_id]
        plt.scatter([bx], [by], s=60, marker='s')
        plt.annotate("BEST", (bx, by))
    plt.gca().invert_yaxis()
    plt.xlabel("Rang (layout)")
    plt.ylabel("Génération")
    plt.title(f"Généalogie (mode={mode}, max_edges={max_edges})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flowers", default="flowers.csv", help="Chemin du CSV des fleurs (colonnes x,y)")
    ap.add_argument("--generations", type=int, default=600)
    ap.add_argument("--pop-size", type=int, default=100)
    ap.add_argument("--selection", choices=["tournament","roulette"], default="tournament")
    ap.add_argument("--tournament-k", type=int, default=3)
    ap.add_argument("--crossover", choices=["OX","PMX"], default="OX")
    ap.add_argument("--mutation-rate", type=float, default=0.03)
    ap.add_argument("--mutation-op", choices=["inversion","swap"], default="inversion")
    ap.add_argument("--elitism-rate", type=float, default=0.03)
    ap.add_argument("--adaptive-mutation", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results")
    # NEW: genealogy controls
    ap.add_argument("--genealogy", choices=["lineage", "full"], default="lineage",
                    help="Type de graphe de généalogie à générer (lineage=chaîne directe lisible, full=forêt limitée)")
    ap.add_argument("--max-edges", type=int, default=2000,
                    help="Limite d'arêtes en mode 'full' pour éviter les freezes")
    args = ap.parse_args()

    flowers = load_flowers(args.flowers)
    field = Field(flowers=flowers, hive=(500.0, 500.0))

    ga = BeehiveGA(
        field=field,
        pop_size=args.pop_size,
        selection=args.selection,
        tournament_k=args.tournament_k,
        crossover=args.crossover,
        mutation_rate=args.mutation_rate,
        mutation_op=args.mutation_op,
        elitism_rate=args.elitism_rate,
        adaptive_mutation=args.adaptive_mutation,
        seed=args.seed,
    )

    # dossier d'output daté
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.outdir, "runs", f"run-{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # exécuter
    population, hist = ga.run(n_generations=args.generations)
    best_bee = min(population, key=lambda b: b.fitness)

    # sauvegardes
    plot_best_path(field, best_bee, os.path.join(run_dir, "best_path.png"))
    plot_convergence(hist["best_per_gen"], hist["avg_per_gen"], os.path.join(run_dir, "convergence.png"))
    plot_genealogy(
        hist["genealogy"],
        best_bee.id,
        os.path.join(run_dir, "genealogy.png"),
        mode=args.genealogy,
        max_edges=args.max_edges
    )

    # métriques
    metrics = pd.DataFrame({
        "generation": list(range(len(hist["best_per_gen"]))),
        "best_distance": hist["best_per_gen"],
        "avg_distance": hist["avg_per_gen"],
    })
    metrics.to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

    # params
    params = vars(args)
    params_json = json.dumps(params, indent=2, ensure_ascii=False)
    with open(os.path.join(run_dir, "params.json"), "w", encoding="utf-8") as f:
        f.write(params_json)

    # console
    print("=== Résultats ===")
    print(f"Meilleure distance finale: {best_bee.fitness:.2f}")
    print(f"Dossier: {run_dir}")
    print("Fichiers générés: best_path.png, convergence.png, genealogy.png, metrics.csv, params.json")


if __name__ == "__main__":
    main()
