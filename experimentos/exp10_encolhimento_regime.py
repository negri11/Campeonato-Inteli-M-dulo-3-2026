"""Encolhimento do coeficiente de regime: escala da coluna regime (k maior = menos penalizacao)."""
import io, contextlib, json, numpy as np, pandas as pd
with contextlib.redirect_stdout(io.StringIO()):
    from nbenv import ns
g = ns; train, Y, M, X_test = g["train"], g["Y_all"], g["M_all"], g["X_test"]
Modelo = g["Modelo"]; orig = Modelo._design
K = {"k": 1.0}
def design(self, X, fit=False):
    Z = orig(self, X, fit)
    if self.g["regime"]:
        j = self.cols.index("regime")
        Z = Z.copy(); Z[:, j] *= K["k"]
    return Z
Modelo._design = design
gen = json.load(open("artefatos/genomas_finais.json"))["final"]
datas = pd.to_datetime(train.Date); pos = (datas > "2021-09-30").to_numpy(); todas = np.ones(len(train), bool)
def ce(T, P, m): return float(-(T[m] * np.log(np.clip(P[m], 1e-15, 1))).sum(1).mean())
for k in [1, 2, 4, 8, 16]:
    K["k"] = float(k)
    Pw = np.mean([g["oof_predict"](x, g["SPLITS"]["wf"]) for x in gen], axis=0)
    ok = ~np.isnan(Pw[:, 0]); m21 = ok & (train.Season == 2021).to_numpy()
    Pt = np.mean([g["fit_genome"](x, todas).predict_proba(X_test) for x in gen], axis=0)
    Ph = np.full((len(train), 3), np.nan)
    Ph[pos] = np.mean([g["fit_genome"](x, ~pos).predict_proba(g["X_all"][pos]) for x in gen], axis=0)
    print(f"k={k:2d} | wf hard {ce(Y, Pw, ok):.4f} soft {ce(M, Pw, ok):.4f} | 2021 hard {ce(Y, Pw, m21):.4f} soft {ce(M, Pw, m21):.4f} "
          f"| pos-regime hard {ce(Y, Ph, pos):.4f} soft {ce(M, Ph, pos):.4f} | media teste A/D/H {Pt.mean(0).round(4)}")
