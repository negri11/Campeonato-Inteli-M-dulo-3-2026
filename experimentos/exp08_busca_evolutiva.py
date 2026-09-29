"""Busca evolutiva sobre familias de modelos. Meta: chegar o mais perto possivel do LogLoss do mercado."""
import sys, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestRegressor, ExtraTreesRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
DATA = Path(__file__).resolve().parent.parent / "data"
CLASSES = ["A", "D", "H"]
SEED = 42
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
        ok = ok & m
    return float(-(T[ok] * np.log(np.clip(P[ok], 1e-12, 1))).sum(1).mean())


def build_features(df):
    X = pd.DataFrame(index=df.index)
    X["Home"], X["Away"] = df.Home.astype(str), df.Away.astype(str)
    for c in ["elo_home", "elo_away", "form_pts_home", "form_pts_away", "form_gf_home", "form_ga_home",
              "form_gf_away", "form_ga_away", "home_form_as_host", "away_form_as_visitor",
              "season_ppg_home", "season_ppg_away", "h2h_home_pts", "round_n", "elo_diff"]:
        X[c] = df[c]
    X["elo_diff_abs"] = df.elo_diff.abs()
    X["form_pts_diff"] = df.form_pts_home - df.form_pts_away
    X["form_gd_diff"] = (df.form_gf_home - df.form_ga_home) - (df.form_gf_away - df.form_ga_away)
    X["form_goals_total"] = df.form_gf_home + df.form_ga_home + df.form_gf_away + df.form_ga_away
    X["venue_form_diff"] = df.home_form_as_host - df.away_form_as_visitor
    j = (df.round_n - 1).clip(lower=0)
    X["ppg_diff"] = (df.season_ppg_home - df.season_ppg_away) * j / (j + 8)
    X["rest_diff"] = df.rest_days_home.clip(0, 10) - df.rest_days_away.clip(0, 10)
    X["short_rest_home"] = (df.rest_days_home <= 3).astype(float)
    X["short_rest_away"] = (df.rest_days_away <= 3).astype(float)
    d = pd.to_datetime(df.Date)
    X["midweek"] = d.dt.dayofweek.isin([1, 2, 3]).astype(float)
    X["month"] = d.dt.month.astype(float)
    X["regime"] = ((d >= "2020-08-01") & (d <= "2021-09-30")).astype(float)
    return X


X = build_features(train)

# grupos de features: (versao bruta, versao em diferenca)
GROUPS = {
    "elo_abs": (["elo_diff_abs"], ["elo_diff_abs"]),
    "form": (["form_pts_home", "form_pts_away"], ["form_pts_diff"]),
    "gols": (["form_gf_home", "form_ga_home", "form_gf_away", "form_ga_away"], ["form_gd_diff", "form_goals_total"]),
    "mando": (["home_form_as_host", "away_form_as_visitor"], ["venue_form_diff"]),
    "ppg": (["season_ppg_home", "season_ppg_away"], ["ppg_diff"]),
    "descanso": (["rest_diff", "short_rest_home", "short_rest_away"],) * 2,
    "h2h": (["h2h_home_pts"],) * 2,
    "rodada": (["round_n"],) * 2,
    "calendario": (["midweek", "month"],) * 2,
    "regime": (["regime"],) * 2,
}
FAMILIES = ["logit", "ordinal", "ridge", "mlp", "hgb", "rf", "et", "knn"]
GENES = {
    "family": FAMILIES,
    "repr": ["bruto", "diff"],
    "team_fx": [True, False],
    "min_freq": [0, 20, 39],
    "alpha": [0.0, 0.5, 0.8, 1.0],
    "hl": [0, 8, 4, 2],
    "depth": [2, 3, 4, 6],
    "leaf": [20, 40, 80, 120, 200],
    "lr": [0.01, 0.02, 0.03, 0.05, 0.1],
    "n_iter": [100, 200, 300, 400],
    "hidden": [4, 8, 16],
}
for g in GROUPS:
    GENES["g_" + g] = [True, False]
LOGC = (-2.0, 0.7)


def columns(g):
    base = ["elo_home", "elo_away"] if g["repr"] == "bruto" else ["elo_diff"]
    i = 0 if g["repr"] == "bruto" else 1
    for name, versions in GROUPS.items():
        if g["g_" + name]:
            base = base + versions[i]
    return base


class Modelo:
    def __init__(self, g):
        self.g, self.cols = g, columns(g)

    def _design(self, X, fit=False):
        g = self.g
        if fit:
            self.imp = SimpleImputer(strategy="median").fit(X[self.cols])
        Z = self.imp.transform(X[self.cols])
        if fit:
            self.sc = StandardScaler().fit(Z)
        Z = self.sc.transform(Z)
        if g["team_fx"]:
            if fit:
                if g["min_freq"]:
                    self.ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=g["min_freq"], sparse_output=False)
                else:
                    self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                self.ohe.fit(X[["Home", "Away"]])
            Z = np.hstack([Z, self.ohe.transform(X[["Home", "Away"]])])
        return Z

    def fit(self, X, P, w=None):
        g, fam = self.g, self.g["family"]
        Z = self._design(X, True)
        n = len(Z)
        w = np.ones(n) if w is None else np.asarray(w, float)
        C = 10 ** g["logC"]
        if fam in ("logit", "hgb"):
            wr = P.T.reshape(-1) * np.tile(w, 3)
            keep = wr > 0
            if fam == "logit":
                self.m = LogisticRegression(C=C, max_iter=3000)
            else:
                self.m = HistGradientBoostingClassifier(max_depth=g["depth"], learning_rate=g["lr"], max_iter=g["n_iter"],
                                                        min_samples_leaf=g["leaf"], l2_regularization=1.0, random_state=SEED)
            self.m.fit(np.vstack([Z] * 3)[keep], np.repeat(np.arange(3), n)[keep], sample_weight=wr[keep])
        elif fam == "ordinal":
            t1, t2 = P[:, 1] + P[:, 2], P[:, 2]
            Zs, ys, ws = [], [], []
            for thr, t in [(0.0, t1), (1.0, t2)]:
                Zt = np.hstack([Z, np.full((n, 1), thr)])
                for label, wt in [(1, t), (0, 1 - t)]:
                    Zs.append(Zt); ys.append(np.full(n, label)); ws.append(wt * w)
            ws = np.concatenate(ws); keep = ws > 0
            self.m = LogisticRegression(C=C, max_iter=5000).fit(np.vstack(Zs)[keep], np.concatenate(ys)[keep], sample_weight=ws[keep])
        elif fam in ("ridge", "mlp"):
            Pc = np.clip(P, 1e-3, None); Pc = Pc / Pc.sum(1, keepdims=True)
            T = np.stack([np.log(Pc[:, 0] / Pc[:, 1]), np.log(Pc[:, 2] / Pc[:, 1])], 1)
            if fam == "ridge":
                self.m = Ridge(alpha=1.0 / C).fit(Z, T, sample_weight=w)
            else:
                self.m = MLPRegressor(hidden_layer_sizes=(g["hidden"],), alpha=1.0 / C, solver="lbfgs",
                                      max_iter=400, random_state=SEED).fit(Z, T)
        elif fam in ("rf", "et"):
            cls = RandomForestRegressor if fam == "rf" else ExtraTreesRegressor
            self.m = cls(n_estimators=150, max_depth=g["depth"] + 2, min_samples_leaf=g["leaf"], max_features=0.5,
                         random_state=SEED, n_jobs=-1).fit(Z, P, sample_weight=w)
        elif fam == "knn":
            self.m = KNeighborsRegressor(n_neighbors=g["leaf"], weights="distance").fit(Z, P)
        return self

    def predict_proba(self, X):
        fam = self.g["family"]
        Z = self._design(X)
        n = len(Z)
        if fam in ("logit", "hgb"):
            P = self.m.predict_proba(Z)
        elif fam == "ordinal":
            g1 = self.m.predict_proba(np.hstack([Z, np.zeros((n, 1))]))[:, 1]
            g2 = self.m.predict_proba(np.hstack([Z, np.ones((n, 1))]))[:, 1]
            P = np.stack([1 - g1, g1 - g2, g2], 1)
        elif fam in ("ridge", "mlp"):
            T = self.m.predict(Z)
            E = np.exp(np.clip(np.stack([T[:, 0], np.zeros(n), T[:, 1]], 1), -6, 6))
            P = E / E.sum(1, keepdims=True)
        else:
            P = self.m.predict(Z)
        P = np.clip(P, 1e-4, None)
        return P / P.sum(1, keepdims=True)


seasons = [int(s) for s in sorted(train.Season.unique())]
SPLITS = {
    "wf": [([s for s in seasons if s < v], [v]) for v in seasons[-5:]],
    "long": [(seasons[:5], seasons[5:])],
    "long2": [(seasons[:4], seasons[4:])],
}
MKT = {k: ce(Y, M, train.Season.isin([s for _, va in sp for s in va]).to_numpy()) for k, sp in SPLITS.items()}
META = float(np.mean(list(MKT.values())))


def target(idx, alpha):
    return alpha * M[idx] + (1 - alpha) * Y[idx]


def oof(g, splits):
    P = np.full((len(train), 3), np.nan)
    alpha = g["alpha"]
    if g["family"] in ("ridge", "mlp"):
        alpha = max(alpha, 0.5)
    for tr_s, va_s in splits:
        tr, va = train.Season.isin(tr_s).to_numpy(), train.Season.isin(va_s).to_numpy()
        w = None if not g["hl"] else (0.5 ** ((train.Season[tr].max() - train.Season[tr]) / g["hl"])).to_numpy()
        P[va] = Modelo(g).fit(X[tr], target(tr, alpha), w).predict_proba(X[va])
    return P


def key(g):
    # so os genes que a familia usa entram na chave de cache
    fam = g["family"]
    used = ["family", "repr", "team_fx", "alpha", "hl"] + ["g_" + n for n in GROUPS]
    if g["team_fx"]:
        used.append("min_freq")
    if fam in ("logit", "ordinal", "ridge", "mlp"):
        used.append("logC")
    if fam == "mlp":
        used.append("hidden")
    if fam == "hgb":
        used += ["depth", "leaf", "lr", "n_iter"]
    if fam in ("rf", "et"):
        used += ["depth", "leaf"]
    if fam == "knn":
        used.append("leaf")
    return json.dumps({k: (round(g[k], 3) if k == "logC" else g[k]) for k in used}, sort_keys=True)


CACHE = {}


def evaluate(g):
    k = key(g)
    if k not in CACHE:
        out = {}
        for name, sp in SPLITS.items():
            P = oof(g, sp)
            out[name], out[name + "_soft"] = ce(Y, P), ce(M, P)
        out["hard"] = float(np.mean([out[n] for n in SPLITS]))
        out["soft"] = float(np.mean([out[n + "_soft"] for n in SPLITS]))
        out["fitness"] = 0.5 * (out["hard"] + out["soft"])
        out["gap_meta"] = out["hard"] - META
        CACHE[k] = out
    return CACHE[k]


def random_genome(rng):
    g = {k: v[rng.integers(len(v))] for k, v in GENES.items()}
    g["logC"] = float(rng.uniform(*LOGC))
    return g


def mutate(g, rng, rate=0.15):
    h = dict(g)
    for k, v in GENES.items():
        if rng.random() < rate:
            h[k] = v[rng.integers(len(v))]
    if rng.random() < 0.5:
        h["logC"] = float(np.clip(h["logC"] + rng.normal(0, 0.3), *LOGC))
    return h


def crossover(a, b, rng):
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in a}


def evolve(pop_size=28, generations=12, elite=4, seed=SEED, verbose=True):
    rng = np.random.default_rng(seed)
    pop = [random_genome(rng) for _ in range(pop_size)]
    # semente: a configuracao v1 (logit, bruto, efeito de time) entra na populacao inicial
    seed_g = dict(random_genome(rng), family="logit", repr="bruto", team_fx=True, min_freq=0, alpha=1.0, hl=4, logC=float(np.log10(0.2)))
    for n in GROUPS:
        seed_g["g_" + n] = n in ("elo_abs", "form", "gols", "mando", "ppg", "descanso", "h2h", "rodada")
    pop[0] = seed_g
    log = []
    for gen in range(generations + 1):
        t0 = time.time()
        scored = []
        for g in pop:
            r = evaluate(g)
            scored.append((r["fitness"], g, r))
            log.append(dict(geracao=gen, **{k: g[k] for k in g}, **r))
        scored.sort(key=lambda t: t[0])
        if verbose:
            fams = pd.Series([g["family"] for _, g, _ in scored]).value_counts().to_dict()
            b = scored[0]
            print(f"ger {gen:2d} | melhor fitness {b[0]:.4f} hard {b[2]['hard']:.4f} soft {b[2]['soft']:.4f} gap {b[2]['gap_meta']:.4f} "
                  f"| mediana {np.median([s[0] for s in scored]):.4f} | {b[1]['family']} | familias {fams} | {time.time() - t0:.0f}s", flush=True)
        if gen == generations:
            break
        new = [g for _, g, _ in scored[:elite]]
        while len(new) < pop_size:
            idx = [min(rng.integers(len(scored), size=3)) for _ in range(2)]   # torneio de 3 (lista ja ordenada)
            child = mutate(crossover(scored[idx[0]][1], scored[idx[1]][1], rng), rng)
            new.append(child)
        pop = new
    return pd.DataFrame(log)


if __name__ == "__main__":
    print("meta (mercado, media dos cortes):", round(META, 4), {k: round(v, 4) for k, v in MKT.items()})
    t0 = time.time()
    log = evolve()
    print("tempo total: %.0fs | avaliacoes unicas: %d" % (time.time() - t0, len(CACHE)))
    log.to_csv(Path(__file__).parent / "evo_log.csv", index=False)
    cols = ["family", "repr", "team_fx", "min_freq", "alpha", "hl", "logC", "depth", "leaf", "lr", "n_iter", "hidden"] + ["g_" + n for n in GROUPS]
    uniq = log.drop_duplicates(subset=["fitness", "hard", "soft"]).sort_values("fitness")
    print(uniq[cols[:8] + ["wf", "long", "long2", "hard", "soft", "fitness", "gap_meta"]].head(15).round(4).to_string())
    print("\nmelhor por familia:")
    print(uniq.loc[uniq.groupby("family").fitness.idxmin()][["family", "repr", "team_fx", "alpha", "hl", "logC", "hard", "soft", "fitness", "gap_meta"]].sort_values("fitness").round(4).to_string())
    print("\ngrupos de features nos 20 melhores (fracao ligada):")
    print(uniq.head(20)[["g_" + n for n in GROUPS]].mean().round(2).to_string())
