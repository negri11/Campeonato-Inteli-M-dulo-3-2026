"""Exploracao nos dados reais: de-vig, deriva do mando, efeitos de time, decaimento."""
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss

pd.set_option("display.width", 220)
DATA = Path(__file__).resolve().parent.parent / "data"
CLASSES = ["A", "D", "H"]
train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
Y = np.stack([(train.Res == c).to_numpy(float) for c in CLASSES], 1)
Q = np.stack([1 / train.odd_visitante, 1 / train.odd_empate, 1 / train.odd_casa], 1)


def devig_basic(Q):
    return Q / Q.sum(1, keepdims=True)


def devig_power(Q):
    lo, hi = np.ones(len(Q)), np.full(len(Q), 3.0)
    for _ in range(60):
        k = (lo + hi) / 2
        s = (Q ** k[:, None]).sum(1)
        lo, hi = np.where(s > 1, k, lo), np.where(s > 1, hi, k)
    P = Q ** ((lo + hi) / 2)[:, None]
    return P / P.sum(1, keepdims=True)


def devig_shin(Q):
    S = Q.sum(1, keepdims=True)
    lo, hi = np.zeros(len(Q)), np.full(len(Q), 0.4)

    def probs(z):
        z = z[:, None]
        return (np.sqrt(z ** 2 + 4 * (1 - z) * Q ** 2 / S) - z) / (2 * (1 - z))
    for _ in range(60):
        z = (lo + hi) / 2
        s = probs(z).sum(1)
        lo, hi = np.where(s > 1, z, lo), np.where(s > 1, hi, z)
    P = probs((lo + hi) / 2)
    return P / P.sum(1, keepdims=True)


def devig_additive(Q):
    P = Q - (Q.sum(1, keepdims=True) - 1) / 3
    P = np.clip(P, 1e-3, None)
    return P / P.sum(1, keepdims=True)


def nll(P, mask=None):
    m = np.ones(len(P), bool) if mask is None else mask
    return float(-(Y[m] * np.log(np.clip(P[m], 1e-12, 1))).sum(1).mean())


def temper(P, T):
    R = np.clip(P, 1e-12, 1) ** (1 / T)
    return R / R.sum(1, keepdims=True)


print("=== (a) de-vig: logloss do mercado no treino inteiro ===")
DV = {"basic": devig_basic(Q), "power": devig_power(Q), "shin": devig_shin(Q), "additive": devig_additive(Q)}
for k, P in DV.items():
    bestT = min(np.arange(0.8, 1.21, 0.02), key=lambda T: nll(temper(P, T)))
    halves = [nll(P, (train.Season <= 2017).to_numpy()), nll(P, (train.Season > 2017).to_numpy())]
    print(f"{k:9s} ll={nll(P):.5f}  13-17={halves[0]:.5f} 18-21={halves[1]:.5f}  melhor T={bestT:.2f} -> {nll(temper(P, bestT)):.5f}")

print("\n=== (b) deriva: media por temporada (mercado power x real) ===")
M = DV["power"]
d = pd.DataFrame({"Season": train.Season, "mH": M[:, 2], "mD": M[:, 1], "mA": M[:, 0],
                  "yH": Y[:, 2], "yD": Y[:, 1], "yA": Y[:, 0]})
print(d.groupby("Season").mean().round(3))


# ------------------------------------------------------------ features
def build_features(df):
    X = pd.DataFrame(index=df.index)
    X["Home"], X["Away"] = df.Home.astype(str), df.Away.astype(str)
    X["elo_diff"] = df.elo_diff
    X["elo_diff_abs"] = df.elo_diff.abs()
    X["elo_mean"] = (df.elo_home + df.elo_away) / 2
    X["form_pts_diff"] = df.form_pts_home - df.form_pts_away
    X["form_gd_diff"] = (df.form_gf_home - df.form_ga_home) - (df.form_gf_away - df.form_ga_away)
    X["form_goals_total"] = df.form_gf_home + df.form_ga_home + df.form_gf_away + df.form_ga_away
    X["venue_form_diff"] = df.home_form_as_host - df.away_form_as_visitor
    j = (df.round_n - 1).clip(lower=0)
    X["ppg_diff"] = (df.season_ppg_home - df.season_ppg_away) * j / (j + 8)
    X["rest_diff"] = df.rest_days_home.clip(0, 10) - df.rest_days_away.clip(0, 10)
    X["short_rest_home"] = (df.rest_days_home <= 3).astype(float)
    X["short_rest_away"] = (df.rest_days_away <= 3).astype(float)
    X["h2h_home_pts"] = df.h2h_home_pts
    X["round_n"] = df.round_n
    return X


FS = {
    "elo": ["elo_diff"],
    "forca": ["elo_diff", "elo_diff_abs", "ppg_diff", "form_pts_diff", "form_gd_diff"],
    "tudo": ["elo_diff", "elo_diff_abs", "elo_mean", "ppg_diff", "form_pts_diff", "form_gd_diff",
             "form_goals_total", "venue_form_diff", "h2h_home_pts", "rest_diff",
             "short_rest_home", "short_rest_away", "round_n"],
}
X = build_features(train)


class SoftLogit:
    def __init__(self, cols, C, team=None, team_scale=1.0):
        self.cols, self.C, self.team, self.ts = cols, C, team, team_scale

    def _design(self, X, fit=False):
        if fit:
            self.imp = SimpleImputer(strategy="median").fit(X[self.cols])
        Z = self.imp.transform(X[self.cols])
        if fit:
            self.sc = StandardScaler().fit(Z)
        Z = self.sc.transform(Z)
        if self.team:
            cols = {"both": ["Home", "Away"], "home": ["Home"], "away": ["Away"]}[self.team]
            if fit:
                self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X[cols])
            Z = np.hstack([Z, self.ts * self.ohe.transform(X[cols])])
        return Z

    def fit(self, X, P, w=None):
        Z = self._design(X, True)
        n = len(Z)
        wr = P.T.reshape(-1).copy()
        if w is not None:
            wr *= np.tile(w, 3)
        keep = wr > 0
        self.clf = LogisticRegression(C=self.C, max_iter=3000)
        self.clf.fit(np.vstack([Z] * 3)[keep], np.repeat(np.arange(3), n)[keep], sample_weight=wr[keep])
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(self._design(X))


seasons = sorted(train.Season.unique())
WF = [([s for s in seasons if s < v], [v]) for v in seasons[-5:]]
LONG = [(seasons[:5], seasons[5:])]
LONG2 = [(seasons[:4], seasons[4:])]


def oof(cfg, splits, tgt):
    P = np.full((len(train), 3), np.nan)
    for tr_s, va_s in splits:
        tr, va = train.Season.isin(tr_s).to_numpy(), train.Season.isin(va_s).to_numpy()
        w = None if cfg["hl"] is None else (0.5 ** ((train.Season[tr].max() - train.Season[tr]) / cfg["hl"])).to_numpy()
        m = SoftLogit(FS[cfg["fs"]], cfg["C"], cfg["team"], cfg.get("ts", 1.0)).fit(X[tr], tgt[tr], w)
        P[va] = m.predict_proba(X[va])
    return P


def ev(cfg, tgt):
    Pw, Pl, Pl2 = oof(cfg, WF, tgt), oof(cfg, LONG, tgt), oof(cfg, LONG2, tgt)
    mw, ml, ml2 = ~np.isnan(Pw[:, 0]), ~np.isnan(Pl[:, 0]), ~np.isnan(Pl2[:, 0])
    return dict(wf=nll(Pw, mw), long=nll(Pl, ml), long2=nll(Pl2, ml2)), Pw, Pl


base = dict(fs="tudo", C=0.1, team="both", hl=4)

print("\n=== (c) alvo por metodo de de-vig (cfg base) ===")
for k, P in DV.items():
    r, Pw, Pl = ev(base, P)
    mw, ml = ~np.isnan(Pw[:, 0]), ~np.isnan(Pl[:, 0])
    tb = min(np.arange(0.8, 1.21, 0.02), key=lambda T: nll(temper(Pw, T), mw) + nll(temper(Pl, T), ml))
    print(f"{k:9s}", {a: round(b, 4) for a, b in r.items()}, "melhor T:", round(tb, 2))

tgt = DV["power"]
print("\n=== (d) efeitos de time ===")
for team, ts in [(None, 1), ("home", 1), ("away", 1), ("both", 0.5), ("both", 1), ("both", 1.5), ("both", 2.5)]:
    r, _, _ = ev(dict(base, team=team, ts=ts), tgt)
    print(f"team={str(team):5s} ts={ts:<4}", {a: round(b, 4) for a, b in r.items()})

print("\n=== (e) decaimento temporal ===")
for hl in [None, 8, 4, 2, 1]:
    r, _, _ = ev(dict(base, hl=hl), tgt)
    print(f"hl={str(hl):5s}", {a: round(b, 4) for a, b in r.items()})

print("\n=== (f) C x features (power, team both, hl 4) ===")
for fs, C in itertools.product(FS, [0.03, 0.1, 0.3, 1.0]):
    r, _, _ = ev(dict(base, fs=fs, C=C), tgt)
    print(f"{fs:6s} C={C:<5}", {a: round(b, 4) for a, b in r.items()})

print("\n=== (g) quao bem o modelo reproduz o mercado fora da amostra ===")
_, Pw, _ = ev(base, tgt)
mw = ~np.isnan(Pw[:, 0])
for j, c in enumerate(CLASSES):
    print(c, "corr modelo x mercado:", round(np.corrcoef(Pw[mw, j], tgt[mw, j])[0, 1], 3),
          "| media modelo", round(Pw[mw, j].mean(), 3), "mercado", round(tgt[mw, j].mean(), 3), "real", round(Y[mw, j].mean(), 3))
