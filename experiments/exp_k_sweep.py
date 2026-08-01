"""Thin entry point: run the k_sweep study. See experiments/run_all.py."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.run_all import STUDIES
if __name__ == "__main__":
    profile = sys.argv[1] if len(sys.argv) > 1 else "reduced"
    STUDIES["k_sweep"](profile)
