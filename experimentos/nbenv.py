"""Carrega as definicoes do caderno (sem rodar as celulas pesadas) para uso em scripts de analise."""
import os, nbformat
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.as_posix()
os.chdir(ROOT)
import matplotlib
matplotlib.use("Agg")
nb = nbformat.read(ROOT + "/brasileirao_probabilidades.ipynb", as_version=4)
SKIP = ["ablacoes = {", "g_com, g_sem", "g_pre = dict", "pos = (datas", "unicos = evo_log", "TOP_K, MAX_POR_FAMILIA",
        "P_val = temper", "todas = np.ones", "amostra = rng.permutation", "def write_submission", "variantes = {"]
ns = {}
for c in nb.cells:
    if c.cell_type != "code":
        continue
    src = c.source
    if any(s in src for s in SKIP):
        continue
    if "def evolve(" in src:
        src = src.split("t0 = time.time()\nevo_log")[0]
    exec(src, ns)
