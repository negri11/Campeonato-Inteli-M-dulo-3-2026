"""Ganho dos efeitos de time em funcao do horizonte (anos depois do fim do treino)."""
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import exp02_features_blends as d
import numpy as np, pandas as pd
train, X, M, Y, FS, Model, ce = d.train, d.X, d.M, d.Y, d.FS, d.Model, d.ce
rows = []
for last in [2015, 2016, 2017, 2018]:
    tr = (train.Season <= last).to_numpy()
    w = (0.5 ** ((last - train.Season[tr]) / 4)).to_numpy()
    m1 = Model(FS["tudo"], 0.1, team=True).fit(X[tr], M[tr], w)
    m0 = Model(FS["tudo"], 0.1, team=False).fit(X[tr], M[tr], w)
    for s in range(last + 1, 2022):
        va = (train.Season == s).to_numpy()
        P1, P0 = m1.predict_proba(X[va]), m0.predict_proba(X[va])
        Pm = 0.5 * P1 + 0.5 * P0
        mk = np.ones(va.sum(), bool)
        rows.append(dict(treino_ate=last, h=s - last,
                         soft_gain=ce(M[va], P0, mk) - ce(M[va], P1, mk),
                         hard_gain=ce(Y[va], P0, mk) - ce(Y[va], P1, mk),
                         soft_gain_half=ce(M[va], P0, mk) - ce(M[va], Pm, mk)))
r = pd.DataFrame(rows)
print(r.round(4).to_string())
print(r.groupby("h")[["soft_gain", "hard_gain", "soft_gain_half"]].mean().round(4))
