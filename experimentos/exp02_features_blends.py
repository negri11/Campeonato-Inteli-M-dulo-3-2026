"""Variantes de features e blends, avaliadas contra resultado real (hard) e contra mercado (soft)."""
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
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


def ce(T, P, m):
    return float(-(T[m] * np.log(np.clip(P[m], 1e-12, 1))).sum(1).mean())


def build_features(df):
    X = pd.DataFrame(index=df.index)
    X["Home"], X["Away"] = df.Home.astype(str), df.Away.astype(str)
    for c in ["elo_home", "elo_away", "form_pts_home", "form_pts_away", "form_gf_home", "form_ga_home",
              "form_gf_away", "form_ga_away", "home_form_as_host", "away_form_as_visitor",
              "season_ppg_home", "season_ppg_away"]:
        X["raw_" + c] = df[c]
    X["elo_diff"] = df.elo_diff
    X["elo_diff_abs"] = df.elo_diff.abs()
    X["elo_diff2"] = (df.elo_diff / 100) ** 2
    X["elo_diff3"] = (df.elo_diff / 100) ** 3
    X["elo_mean"] = (df.elo_home + df.elo_away) / 2
    X["form_pts_diff"] = df.form_pts_home - df.form_pts_away
    X["form_gd_diff"] = (df.form_gf_home - df.form_ga_home) - (df.form_gf_away - df.form_ga_away)
    X["form_goals_total"] = df.form_gf_home + df.form_ga_home + df.form_gf_away + df.form_ga_away
    X["venue_form_diff"] = df.home_form_as_host - df.away_form_as_visitor
    j = (df.round_n - 1).clip(lower=0)
    X["ppg_diff"] = (df.season_ppg_home - df.season_ppg_away) * j / (j + 8)
    X["ppg_diff_abs"] = X["ppg_diff"].abs()
    X["ppg_sum"] = (df.season_ppg_home + df.season_ppg_away) * j / (j + 8)
    X["rest_diff"] = df.rest_days_home.clip(0, 10) - df.rest_days_away.clip(0, 10)
    X["short_rest_home"] = (df.rest_days_home <= 3).astype(float)
    X["short_rest_away"] = (df.rest_days_away <= 3).astype(float)
    X["long_rest"] = ((df.rest_days_home > 30) | (df.rest_days_away > 30)).astype(float)
    X["h2h_home_pts"] = df.h2h_home_pts
    X["h2h_missing"] = df.h2h_home_pts.isna().astype(float)
    X["round_n"] = df.round_n
    X["late"] = (df.round_n >= 30).astype(float)
    X["elo_x_round"] = df.elo_diff * (df.round_n - 19) / 19
    X["ppg_x_round"] = X["ppg_diff"] * (df.round_n - 19) / 19
    return X


TUDO = ["elo_diff", "elo_diff_abs", "elo_mean", "ppg_diff", "form_pts_diff", "form_gd_diff",
        "form_goals_total", "venue_form_diff", "h2h_home_pts", "rest_diff",
        "short_rest_home", "short_rest_away", "round_n"]
FS = {
    "elo": ["elo_diff"],
    "elo+ppg": ["elo_diff", "ppg_diff"],
    "forca": ["elo_diff", "elo_diff_abs", "ppg_diff", "form_pts_diff", "form_gd_diff"],
    "tudo": TUDO,
    "tudo-abs": [c for c in TUDO if c != "elo_diff_abs"],
    "tudo+poly": TUDO + ["elo_diff2", "elo_diff3"],
    "tudo+inter": TUDO + ["elo_x_round", "ppg_x_round"],
    "tudo+extra": TUDO + ["ppg_diff_abs", "ppg_sum", "long_rest", "h2h_missing", "late"],
    "raw": [c for c in build_features(train.head(2)).columns if c.startswith("raw_")] +
           ["elo_diff_abs", "rest_diff", "h2h_home_pts", "round_n"],
    "kitchen": TUDO + ["elo_diff2", "elo_diff3", "elo_x_round", "ppg_x_round", "ppg_diff_abs", "ppg_sum",
                       "long_rest", "h2h_missing", "late"],
}
X = build_features(train)


class Model:
    def __init__(self, cols, C=0.1, team=True, kind="logit", hgb=None):
        self.cols, self.C, self.team, self.kind, self.hgb = cols, C, team, kind, hgb or {}

    def _design(self, X, fit=False):
        if fit:
            self.imp = SimpleImputer(strategy="median").fit(X[self.cols])
        Z = self.imp.transform(X[self.cols])
        if fit:
            self.sc = StandardScaler().fit(Z)
        Z = self.sc.transform(Z)
        if self.team and self.kind == "logit":
            if fit:
                self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X[["Home", "Away"]])
            Z = np.hstack([Z, self.ohe.transform(X[["Home", "Away"]])])
        return Z

    def fit(self, X, P, w=None):
        Z = self._design(X, True)
        n = len(Z)
        wr = P.T.reshape(-1).copy()
        if w is not None:
            wr *= np.tile(w, 3)
        keep = wr > 0
        if self.kind == "logit":
            self.clf = LogisticRegression(C=self.C, max_iter=3000)
        else:
            self.clf = HistGradientBoostingClassifier(random_state=0, **self.hgb)
        self.clf.fit(np.vstack([Z] * 3)[keep], np.repeat(np.arange(3), n)[keep], sample_weight=wr[keep])
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(self._design(X))


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


def report(name, make, keep=None):
    out = {}
    for k, sp in SPLITS.items():
        P = oof(make, sp)
        m = ~np.isnan(P[:, 0])
        out[k] = ce(Y, P, m)
        out[k + "_soft"] = ce(M, P, m)
        if keep is not None:
            keep[(name, k)] = P
    out["hard"] = np.mean([out[k] for k in SPLITS])
    out["soft"] = np.mean([out[k + "_soft"] for k in SPLITS])
    print(f"{name:28s}", " ".join(f"{k}={v:.4f}" for k, v in out.items()))
    return out


print("piso do soft (entropia do proprio mercado):",
      {k: round(ce(M, M, train.Season.isin([s for _, v in sp for s in v]).to_numpy()), 4) for k, sp in SPLITS.items()})

store = {}
print("\n=== features (logit, team fx, C=0.1) ===")
for fs in FS:
    report(fs, lambda fs=fs: Model(FS[fs], 0.1), store)

print("\n=== C fino em 'tudo' e 'kitchen' ===")
for fs, C in itertools.product(["tudo", "kitchen"], [0.05, 0.1, 0.2, 0.3, 0.5]):
    report(f"{fs} C={C}", lambda fs=fs, C=C: Model(FS[fs], C), store)

print("\n=== HGB alvo suave ===")
for name, hp in {
    "hgb d2 lr.03 n150": dict(max_depth=2, learning_rate=0.03, max_iter=150, min_samples_leaf=60, l2_regularization=1.0),
    "hgb d2 lr.03 n300": dict(max_depth=2, learning_rate=0.03, max_iter=300, min_samples_leaf=60, l2_regularization=1.0),
    "hgb d3 lr.03 n200": dict(max_depth=3, learning_rate=0.03, max_iter=200, min_samples_leaf=100, l2_regularization=1.0),
}.items():
    report(name, lambda hp=hp: Model(FS["tudo"], kind="hgb", hgb=hp), store)

print("\n=== blends logit(tudo C=0.1) + hgb ===")
for hname in ["hgb d2 lr.03 n300", "hgb d3 lr.03 n200"]:
    for w in [0.1, 0.2, 0.3, 0.5]:
        out = {}
        for k in SPLITS:
            P = (1 - w) * store[("tudo C=0.1", k)] + w * store[(hname, k)]
            m = ~np.isnan(P[:, 0])
            out[k], out[k + "_soft"] = ce(Y, P, m), ce(M, P, m)
        print(f"w_hgb={w} {hname:20s}", " ".join(f"{k}={v:.4f}" for k, v in out.items()),
              f"hard={np.mean([out[k] for k in SPLITS]):.4f} soft={np.mean([out[k + '_soft'] for k in SPLITS]):.4f}")
