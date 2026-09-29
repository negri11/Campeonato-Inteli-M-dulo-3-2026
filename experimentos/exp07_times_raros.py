"""Times raros/ineditos: agrupar em categoria unica via min_frequency."""
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import exp04_ordinal_calendario as d
import numpy as np, pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
train, X, M, Y, ce = d.train, d.X, d.M, d.Y, d.ce
dt = pd.to_datetime(train.Date)
X["regime"] = ((dt >= "2020-08-01") & (dt <= "2021-09-30")).astype(float)
COLS = d.FS["bruto"] + ["regime"]

class MultiMF(d.Multi):
    def __init__(self, cols, C=0.2, min_freq=None):
        super().__init__(cols, C, True); self.min_freq = min_freq
    def _design(self, X, fit=False):
        if fit:
            self.imp = SimpleImputer(strategy="median").fit(X[self.cols])
        Z = self.imp.transform(X[self.cols])
        if fit:
            self.sc = StandardScaler().fit(Z)
        Z = self.sc.transform(Z)
        if fit:
            if self.min_freq:
                self.ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=self.min_freq, sparse_output=False)
            else:
                self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.ohe.fit(X[["Home", "Away"]])
        return np.hstack([Z, self.ohe.transform(X[["Home", "Away"]])])

def run(make, hl=4, mask_fn=None):
    out = {}
    for k, sp in d.SPLITS.items():
        P = np.full((len(train), 3), np.nan)
        for tr_s, va_s in sp:
            a, b = train.Season.isin(tr_s).to_numpy(), train.Season.isin(va_s).to_numpy()
            w = (0.5 ** ((train.Season[a].max() - train.Season[a]) / hl)).to_numpy()
            P[b] = make().fit(X[a], M[a], w).predict_proba(X[b])
            if mask_fn is not None:
                known = set(train.Home[a])
                novo = (~train.Home.isin(known) | ~train.Away.isin(known)).to_numpy()
                P[b & ~novo] = np.nan
        out[k], out[k + "_s"] = ce(Y, P), ce(M, P)
        out[k + "_n"] = int((~np.isnan(P[:, 0])).sum())
    return out

for mf in [None, 20, 39, 58]:
    r = run(lambda mf=mf: MultiMF(COLS, 0.2, mf))
    print(f"min_freq={str(mf):5s} todas     ", " ".join(f"{k}={v:.4f}" for k, v in r.items() if not k.endswith("_n")))
for mf in [None, 20, 39, 58]:
    r = run(lambda mf=mf: MultiMF(COLS, 0.2, mf), mask_fn=True)
    print(f"min_freq={str(mf):5s} so ineditos", " ".join(f"{k}={v:.4f}" for k, v in r.items() if not k.endswith("_n")), "n:", [r[k + "_n"] for k in d.SPLITS])
