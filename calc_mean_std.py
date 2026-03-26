import os
import argparse
import numpy as np
import re

def parse_result_line(line):
    pattern = r"([\w]+):([\d\.]+)"
    matches = re.findall(pattern, line)
    return {k.lower(): float(v) for k, v in matches}

def load_result(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        line = f.readline().strip()
        return parse_result_line(line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    results = []
    for i in range(args.runs):
        path = f"./results/{args.dataset}_{args.model}_{i}.txt"
        r = load_result(path)
        if r is None:
            print(f"[WARN] Result missing: {path}")
        else:
            print(f"[INFO] Loaded: {path} -> {r}")
            results.append(r)
    if len(results) == 0:
        print("No valid results found.")
        return
    keys = ["accuracy", "precision", "recall", "f1", "avg"]
    stats = {}
    for k in keys:
        vals = np.array([r[k] for r in results])
        stats[k] = (vals.mean(), vals.std())
    print("\n========== FINAL RESULTS (MEAN % ± STD %) ==========")
    for k in keys:
        mean, std = stats[k]
        print(f"{k.capitalize():<10}: {mean*100:.2f}% ± {std*100:.2f}%")
    print("====================================================")

if __name__ == "__main__":
    main()
