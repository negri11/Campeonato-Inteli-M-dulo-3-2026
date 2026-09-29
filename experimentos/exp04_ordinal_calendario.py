"""Rodada 2: modelo ordinal (limiares empilhados), features de calendario, inclinacao do Elo por temporada."""
import io, contextlib, itertools
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

pd.set_option("display.width", 220)
DATA = Path(__file__).resolve().parent.parent / "data"
CLASSES = ["A", "D", "H"]
train = pd.read_csv(DATA / "train.csv")
Y = np.stack([(train.Res == c).to_numpy(float) for c in CLASSES], 1)
Q = np.stack([1 / train.odd_visitante, 1 / train.odd_empate, 1 / train.odd_casa], 1)


def devig_power(Q):
    lo, hi = np.ones(len(Q)), np.full(len(Q), 3.0)
    for _ in range(60):
        k = (lo + hi) / 2
        s = (Q ** k[:, None]).sum(1)
        lo, hi = np.where(s > 1, k, lo), np.where(s > 1, hi, k)
    P = Q ** ((lo + hi) / 2)[:, None]
    return P / P.sum(1, keepdims=True)


M = devig_power(Q)


def ce(T, P, m=None):
    ok = ~np.isnan(P[:, 0])
    if m is not None:
        ok &= m
    return float(-(T[ok] * np.log(np.clip(P[ok], 1e-12, 1))).sum(1).mean())


RAW = ["elo_home", "elo_away", "form_pts_home", "form_pts_away", "form_gf_home", "form_ga_home",
       "form_gf_away", "form_ga_away", "home_form_as_host", "away_form_as_visitor",
       "season_ppg_home", "season_ppg_away"]


def build_features(df):
    X = pd.DataFrame(index=df.index)
    X["Home"], X["Away"] = df.Home.astype(str), df.Away.astype(str)
    for c in RAW:
        X[c] = df[c]
    X["elo_diff"] = df.elo_diff
    X["elo_diff_abs"] = df.elo_diff.abs()
    X["rest_diff"] = df.rest_days_home.clip(0, 10) - df.rest_days_away.clip(0, 10)
    X["rest_home"] = df.rest_days_home.clip(0, 10)
    X["rest_away"] = df.rest_days_away.clip(0, 10)
    X["short_rest_home"] = (df.rest_days_home <= 3).astype(float)
    X["short_rest_away"] = (df.rest_days_away <= 3).astype(float)
    X["h2h_home_pts"] = df.h2h_home_pts
    X["round_n"] = df.round_n
    d = pd.to_datetime(df.Date)
    X["midweek"] = d.dt.dayofweek.isin([1, 2, 3]).astype(float)
    X["month"] = d.dt.month.astype(float)
    X["first_round"] = (df.round_n <= 1).astype(float)
    X["new_home"] = (df.form_pts_home.isna() | (df.rest_days_home > 300)).astype(float)
    X["new_away"] = (df.form_pts_away.isna() | (df.rest_days_away > 300)).astype(float)
    j = (df.round_n - 1).clip(lower=0)
    X["ppg_diff"] = (df.season_ppg_home - df.season_ppg_away) * j / (j + 8)
    X["late_x_ppg_home"] = (df.round_n >= 28) * df.season_ppg_home.fillna(1.35)
    X["late_x_ppg_away"] = (df.round_n >= 28) * df.season_ppg_away.fillna(1.35)
    return X


BRUTO = RAW + ["elo_diff_abs", "rest_diff", "h2h_home_pts", "round_n"]
FS = {
    "bruto": BRUTO,
    "bruto+cal": BRUTO + ["midweek", "month"],
    "bruto+novo": BRUTO + ["new_home", "new_away", "first_round"],
    "bruto+rest": BRUTO + ["rest_home", "rest_away", "short_rest_home", "short_rest_away"],
    "bruto+late": BRUTO + ["late_x_ppg_home", "late_x_ppg_away"],
    "bruto+all": BRUTO + ["midweek", "month", "new_home", "new_away", "first_round", "rest_home", "rest_away",
                          "short_rest_home", "short_rest_away", "late_x_ppg_home", "late_x_ppg_away"],
    "bruto-abs": [c for c in BRUTO if c != "elo_diff_abs"],
}
X = build_features(train)


class Base:
    def __init__(self, cols, C=0.2, team=True):
        self.cols, self.C, self.team = cols, C, team

    def _design(self, X, fit=False):
        if fit:
            self.imp = SimpleImputer(strategy="median").fit(X[self.cols])
        Z = self.imp.transform(X[self.cols])
        if fit:
            self.sc = StandardScaler().fit(Z)
        Z = self.sc.transform(Z)
        if self.team:
            if fit:
                self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X[["Home", "Away"]])
            Z = np.hstack([Z, self.ohe.transform(X[["Home", "Away"]])])
        return Z


class Multi(Base):
    def fit(self, X, P, w=None):
        Z = self._design(X, True)
        n = len(Z)
        wr = P.T.reshape(-1).copy()
        if w is not None:
            wr *= np.tile(w, 3)
        self.clf = LogisticRegression(C=self.C, max_iter=3000)
        self.clf.fit(np.vstack([Z] * 3), np.repeat(np.arange(3), n), sample_weight=wr)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(self._design(X))


class Ordinal(Base):
    """Logit de odds proporcionais via limiares empilhados:
    P(y>A)=sig(xb-c1), P(y>D)=sig(xb-c2), com b compartilhado."""

    def fit(self, X, P, w=None):
        Z = self._design(X, True)
        n = len(Z)
        w = np.ones(n) if w is None else w
        t1, t2 = P[:, 1] + P[:, 2], P[:, 2]          # alvos suaves dos dois limiares
        blocks, ys, ws = [], [], []
        for thr, t in [(0.0, t1), (1.0, t2)]:
            Zt = np.hstack([Z, np.full((n, 1), thr)])
            for label, wt in [(1, t), (0, 1 - t)]:
                blocks.append(Zt); ys.append(np.full(n, label)); ws.append(wt * w)
        self.clf = LogisticRegression(C=self.C, max_iter=5000)
        self.clf.fit(np.vstack(blocks), np.concatenate(ys), sample_weight=np.concatenate(ws))
        return self

    def predict_proba(self, X):
        Z = self._design(X)
        n = len(Z)
        g1 = self.clf.predict_proba(np.hstack([Z, np.zeros((n, 1))]))[:, 1]
        g2 = self.clf.predict_proba(np.hstack([Z, np.ones((n, 1))]))[:, 1]
        P = np.stack([1 - g1, g1 - g2, g2], 1)
        P = np.clip(P, 1e-6, None)
        return P / P.sum(1, keepdims=True)


class LogitRidge(Base):
    """Regressao ridge nos log-odds do mercado (log pH/pD e log pA/pD)."""

    def fit(self, X, P, w=None):
        Z = self._design(X, True)
        T = np.stack([np.log(P[:, 0] / P[:, 1]), np.log(P[:, 2] / P[:, 1])], 1)
        self.reg = Ridge(alpha=1.0 / self.C).fit(Z, T, sample_weight=w)
        return self

    def predict_proba(self, X):
        T = self.reg.predict(self._design(X))
        E = np.exp(np.stack([T[:, 0], np.zeros(len(T)), T[:, 1]], 1))
        return E / E.sum(1, keepdims=True)


seasons = sorted(train.Season.unique())
SPLITS = {
    "wf": [([s for s in seasons if s < v], [v]) for v in seasons[-5:]],
    "long": [(seasons[:5], seasons[5:])],
    "long2": [(seasons[:4], seasons[4:])],
}


def oof(make, splits, hl=4):
    P = np.full((len(train), 3), np.nan)
    for tr_s, va_s in splits:
        tr, va = train.Season.isin(tr_s).to_numpy(), train.Season.isin(va_s).to_numpy()
        w = None if hl is None else (0.5 ** ((train.Season[tr].max() - train.Season[tr]) / hl)).to_numpy()
        P[va] = make().fit(X[tr], M[tr], w).predict_proba(X[va])
    return P


store = {}


def report(name, make, hl=4):
    out = {}
    for k, sp in SPLITS.items():
        P = oof(make, sp, hl)
        out[k], out[k + "_s"] = ce(Y, P), ce(M, P)
        store[(name, k)] = P
    out["hard"] = np.mean([out[k] for k in SPLITS])
    out["soft"] = np.mean([out[k + "_s"] for k in SPLITS])
    print(f"{name:26s}", " ".join(f"{k}={v:.4f}" for k, v in out.items()))
    return out


if __name__ == "__main__":
    print("=== inclinacao do elo_diff por temporada (logit binario H vs resto, alvo mercado) ===")
    for s in seasons:
        m = (train.Season == s).to_numpy()
        z = (train.elo_diff[m] / 100).to_numpy().reshape(-1, 1)
        n = m.sum()
        clf = LogisticRegression(C=1e4, max_iter=1000).fit(
            np.vstack([z, z]), np.r_[np.ones(n), np.zeros(n)], sample_weight=np.r_[M[m, 2], 1 - M[m, 2]])
        print(s, "coef/100pts:", round(clf.coef_[0, 0], 3), "intercepto:", round(clf.intercept_[0], 3),
              "sd elo_diff:", round(train.elo_diff[m].std(), 1))

    print("\n=== features de calendario e afins (multinomial, C=0.2, team) ===")
    for fs in FS:
        report(fs, lambda fs=fs: Multi(FS[fs], 0.2))

    print("\n=== ordinal x multinomial x ridge-logit ===")
    for C in [0.05, 0.1, 0.2, 0.5, 1.0]:
        report(f"ordinal bruto C={C}", lambda C=C: Ordinal(FS["bruto"], C))
    for C in [0.05, 0.1, 0.2, 0.5]:
        report(f"ordinal bruto-abs C={C}", lambda C=C: Ordinal(FS["bruto-abs"], C))
    for C in [0.01, 0.03, 0.1, 0.3]:
        report(f"ridge bruto C={C}", lambda C=C: LogitRidge(FS["bruto"], C))

    print("\n=== blends multinomial + ordinal ===")
    for a, b in [("bruto", "ordinal bruto C=0.2"), ("bruto", "ordinal bruto C=0.5"), ("bruto", "ridge bruto C=0.03")]:
        for w in [0.3, 0.5, 0.7]:
            out = {}
            for k in SPLITS:
                P = (1 - w) * store[(a, k)] + w * store[(b, k)]
                out[k], out[k + "_s"] = ce(Y, P), ce(M, P)
            print(f"w={w} {b:24s}", " ".join(f"{k}={v:.4f}" for k, v in out.items()),
                  f"hard={np.mean([out[k] for k in SPLITS]):.4f} soft={np.mean([out[k + '_s'] for k in SPLITS]):.4f}")
