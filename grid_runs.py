import os
import subprocess
import json
import pandas as pd
from pathlib import Path

# Dossier de résultats
BASE_RESULTS = Path("grid_results")
BASE_RESULTS.mkdir(exist_ok=True)

# Commande de base
BASE_CMD = [
    "python", "main.py",
    "--flowers", "flowers.csv",
    "--generations", "600",
    "--genealogy", "lineage",
    "--seed", "42"
]

# Expériences à tester
experiments = {
    "mut-001": ["--pop-size", "100", "--selection", "tournament", "--tournament-k", "3",
                "--crossover", "OX", "--mutation-rate", "0.01", "--elitism-rate", "0.03"],
    "mut-003": ["--pop-size", "100", "--selection", "tournament", "--tournament-k", "3",
                "--crossover", "OX", "--mutation-rate", "0.03", "--elitism-rate", "0.03"],
    "mut-005": ["--pop-size", "100", "--selection", "tournament", "--tournament-k", "3",
                "--crossover", "OX", "--mutation-rate", "0.05", "--elitism-rate", "0.03"],
    "mut-adapt": ["--pop-size", "100", "--selection", "tournament", "--tournament-k", "3",
                  "--crossover", "OX", "--mutation-rate", "0.03", "--elitism-rate", "0.03", "--adaptive-mutation"],
    "sel-tourn": ["--pop-size", "100", "--selection", "tournament", "--tournament-k", "3",
                  "--crossover", "OX", "--mutation-rate", "0.03", "--elitism-rate", "0.03", "--adaptive-mutation"],
    "sel-roul": ["--pop-size", "100", "--selection", "roulette",
                 "--crossover", "OX", "--mutation-rate", "0.03", "--elitism-rate", "0.03", "--adaptive-mutation"],
    "xo-ox": ["--pop-size", "100", "--selection", "tournament", "--tournament-k", "3",
              "--crossover", "OX", "--mutation-rate", "0.03", "--elitism-rate", "0.03", "--adaptive-mutation"],
    "xo-pmx": ["--pop-size", "100", "--selection", "tournament", "--tournament-k", "3",
               "--crossover", "PMX", "--mutation-rate", "0.03", "--elitism-rate", "0.03", "--adaptive-mutation"],
    "elit-001": ["--pop-size", "100", "--selection", "tournament", "--tournament-k", "3",
                 "--crossover", "OX", "--mutation-rate", "0.03", "--elitism-rate", "0.01", "--adaptive-mutation"],
    "elit-003": ["--pop-size", "100", "--selection", "tournament", "--tournament-k", "3",
                 "--crossover", "OX", "--mutation-rate", "0.03", "--elitism-rate", "0.03", "--adaptive-mutation"],
    "pop-050": ["--pop-size", "50", "--selection", "tournament", "--tournament-k", "3",
                "--crossover", "OX", "--mutation-rate", "0.03", "--elitism-rate", "0.03", "--adaptive-mutation"],
    "pop-100": ["--pop-size", "100", "--selection", "tournament", "--tournament-k", "3",
                "--crossover", "OX", "--mutation-rate", "0.03", "--elitism-rate", "0.03", "--adaptive-mutation"]
}

def run_experiments():
    for name, args in experiments.items():
        print(f"🚀 Running experiment: {name}")
        outdir = BASE_RESULTS / name
        outdir.mkdir(parents=True, exist_ok=True)
        
        # Command with specific outdir
        cmd = BASE_CMD + args + ["--outdir", str(outdir)]
        
        # Run the experiment
        subprocess.run(cmd, check=True)

def collect_results():
    rows = []
    for name in experiments.keys():
        run_dir = BASE_RESULTS / name / "runs"
        if not run_dir.exists():
            continue
        
        # Get most recent run subfolder
        latest = max(run_dir.iterdir(), key=os.path.getmtime)
        
        metrics_path = latest / "metrics.csv"
        params_path = latest / "params.json"
        
        if metrics_path.exists():
            df = pd.read_csv(metrics_path)
            best_final = df["best_distance"].iloc[-1]
            avg_final = df["avg_distance"].iloc[-1]
        else:
            best_final, avg_final = None, None
        
        if params_path.exists():
            with open(params_path) as f:
                params = json.load(f)
        else:
            params = {}
        
        rows.append({
            "experiment": name,
            "best_final": best_final,
            "avg_final": avg_final,
            "params": params
        })
    
    # Save summary
    summary = pd.DataFrame(rows)
    summary.to_csv(BASE_RESULTS / "comparison.csv", index=False)
    print(f"✅ Résultats sauvegardés dans {BASE_RESULTS/'comparison.csv'}")

if __name__ == "__main__":
    run_experiments()
    collect_results()
