"""Feature de regime (mando reduzido 2020-08 a 2021-09) e peso de recencia."""
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import exp04_ordinal_calendario as d
import numpy as np, pandas as pd
train, X, M, Y, ce = d.train, d.X, d.M, d.Y, d.ce
dt = pd.to_datetime(train.Date)
X["regime"] = ((dt >= "2020-08-01") & (dt <= "2021-09-30")).astype(float)
X["regime_x_elo"] = X["regime"] * train.elo_diff / 100
B = d.FS["bruto"]
SETS = {"base": B, "regime": B + ["regime"], "regime+elo": B + ["regime", "regime_x_elo"]}

def fit(cols, idx, hl, C=0.2, regime_off=False):
    w = None if hl is None else (0.5 ** ((train.Season[idx].max() - train.Season[idx]) / hl)).to_numpy()
    return d.Multi(cols, C).fit(X[idx], M[idx], w)

print("=== holdout pos-regime: treino ate 2021-09-30, teste 2021-10 em diante (n=%d) ===" % (dt > "2021-09-30").sum())
tr = (dt <= "2021-09-30").to_numpy(); te = (dt > "2021-09-30").to_numpy()
for name, cols in SETS.items():
    for hl in [None, 4, 2]:
        m = fit(cols, tr, hl)
        P = np.full((len(train), 3), np.nan); P[te] = m.predict_proba(X[te])
        print(f"{name:11s} hl={str(hl):4s} hard={ce(Y, P):.4f} soft={ce(M, P):.4f} mediaH={P[te, 2].mean():.3f} (mercado {M[te, 2].mean():.3f}, real {Y[te, 2].mean():.3f})")

print("\n=== so dezembro/2021 (n=%d) ===" % (dt >= "2021-12-01").sum())
te2 = (dt >= "2021-12-01").to_numpy()
for name, cols in SETS.items():
    for hl in [None, 4]:
        m = fit(cols, tr, hl)
        P = np.full((len(train), 3), np.nan); P[te2] = m.predict_proba(X[te2])
        print(f"{name:11s} hl={str(hl):4s} hard={ce(Y, P):.4f} soft={ce(M, P):.4f} mediaH={P[te2, 2].mean():.3f} (mercado {M[te2, 2].mean():.3f})")

print("\n=== cortes usuais (regime como feature da linha) ===")
seasons = sorted(train.Season.unique())
for name, cols in SETS.items():
    for hl in [None, 4, 2]:
        out = {}
        for k, sp in d.SPLITS.items():
            P = np.full((len(train), 3), np.nan)
            for tr_s, va_s in sp:
                a, b = train.Season.isin(tr_s).to_numpy(), train.Season.isin(va_s).to_numpy()
                P[b] = fit(cols, a, hl).predict_proba(X[b])
            out[k], out[k + "_s"] = ce(Y, P), ce(M, P)
        print(f"{name:11s} hl={str(hl):4s}", " ".join(f"{k}={v:.4f}" for k, v in out.items()))

print("\n=== previsao media no teste com cada variante (treino completo) ===")
test = pd.read_csv(d.DATA / "test.csv")
Xt = d.build_features(test); Xt["regime"] = 0.0; Xt["regime_x_elo"] = 0.0
allm = np.ones(len(train), bool)
for name, cols in SETS.items():
    for hl in [None, 4, 2]:
        P = fit(cols, allm, hl).predict_proba(Xt)
        print(f"{name:11s} hl={str(hl):4s} media A/D/H = {P.mean(0).round(4)}")
m = fit(SETS["regime"], allm, 4)
print("coef regime (A, D, H):", m.clf.coef_[:, len(B)].round(3))
