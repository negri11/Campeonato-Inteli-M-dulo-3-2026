"""Diferenca pareada entre familias (ridge x logit) por temporada, com erro padrao."""
import json, numpy as np, pandas as pd
from nbenv import ns
g = ns; train, Y, M = g["train"], g["Y_all"], g["M_all"]
gen = json.load(open("artefatos/genomas_finais.json"))
ridge, logit = gen["ridge"], gen["logit"]
def ens(gs, splits):
    return np.mean([g["oof_predict"](x, splits) for x in gs], axis=0)
def per_match(T, P):
    return -(T * np.log(np.clip(P, 1e-15, 1))).sum(1)
for name, sp in g["SPLITS"].items():
    Pr, Pl = ens(ridge, sp), ens(logit, sp)
    ok = ~np.isnan(Pr[:, 0])
    d_hard = per_match(Y, Pr) - per_match(Y, Pl)
    d_soft = per_match(M, Pr) - per_match(M, Pl)
    print(f"\n[{name}] ridge - logit | hard {d_hard[ok].mean():+.4f} (EP {d_hard[ok].std() / np.sqrt(ok.sum()):.4f}) | soft {d_soft[ok].mean():+.4f} | dif. abs media de prob {np.abs(Pr[ok] - Pl[ok]).mean():.4f}")
    rows = []
    for s in sorted(train.Season[ok].unique()):
        m = ok & (train.Season == s).to_numpy()
        rows.append(dict(temporada=int(s), hard=d_hard[m].mean(), soft=d_soft[m].mean(), ep_hard=d_hard[m].std() / np.sqrt(m.sum())))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    # blocos de ~517 partidas (tamanho do placar publico)
    idx = np.where(ok)[0]; rng = np.random.default_rng(0)
    sims = [d_hard[rng.choice(idx, 517, replace=False)].mean() for _ in range(2000)]
    print("amostras de 517 partidas: desvio padrao da diferenca hard = %.4f" % np.std(sims))
