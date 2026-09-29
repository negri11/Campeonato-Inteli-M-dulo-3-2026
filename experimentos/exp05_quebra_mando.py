"""Quebra estrutural do mando em 2020-2021: residuo do mercado contra modelo treinado ate 2019."""
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import exp04_ordinal_calendario as d
import numpy as np, pandas as pd
train, X, M, Y = d.train, d.X, d.M, d.Y
tr = (train.Season <= 2019).to_numpy()
m = d.Multi(d.FS["bruto"], 0.2).fit(X[tr], M[tr], None)
P = m.predict_proba(X)
df = pd.DataFrame({"Season": train.Season, "ym": pd.to_datetime(train.Date).dt.to_period("Q"),
                   "mkt_H": M[:, 2], "mod_H": P[:, 2], "mkt_A": M[:, 0], "mod_A": P[:, 0], "y_H": Y[:, 2], "y_A": Y[:, 0]})
df["res_H"] = df.mkt_H - df.mod_H
df["res_A"] = df.mkt_A - df.mod_A
g = df[df.Season >= 2018].groupby("ym").agg(n=("res_H", "size"), mkt_H=("mkt_H", "mean"), mod_H=("mod_H", "mean"),
                                             res_H=("res_H", "mean"), res_A=("res_A", "mean"), y_H=("y_H", "mean"))
print(g.round(3))
d21 = df[train.Season == 2021].copy()
d21["mes"] = pd.to_datetime(train.Date[train.Season == 2021]).dt.to_period("M")
print(d21.groupby("mes").agg(n=("res_H", "size"), mkt_H=("mkt_H", "mean"), mod_H=("mod_H", "mean"), res_H=("res_H", "mean"), y_H=("y_H", "mean")).round(3))
print("datas 2020:", train.Date[train.Season == 2020].min(), train.Date[train.Season == 2020].max())
print("datas 2021:", train.Date[train.Season == 2021].min(), train.Date[train.Season == 2021].max())
