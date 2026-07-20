import pandas as pd, numpy as np
# read a chunk of feature columns to characterize
trf = pd.read_csv("data/train_features.csv", nrows=2000)
feat = trf.drop(columns=[c for c in ['id','label'] if c in trf.columns])
X = feat.to_numpy(dtype=float)
print("shape sample:", X.shape)
print("global min/max/mean:", X.min(), X.max(), round(X.mean(),4))
print("fraction zeros:", round((X==0).mean(),4))
print("fraction integers (val==round):", round((X==np.round(X)).mean(),4))
# per-column: how many are all-integer vs float
col_all_int = (X==np.round(X)).all(axis=0)
print("columns all-integer:", int(col_all_int.sum()), "of", X.shape[1])
# range of non-integer columns
if (~col_all_int).any():
    fcols = X[:, ~col_all_int]
    print("float-col min/max:", round(fcols.min(),4), round(fcols.max(),4))
# integer col max values -> counts?
icols = X[:, col_all_int]
print("int-col max:", icols.max(), "int-col value examples row0:", icols[0,:8])
# are rows L2-normalized? sum of squares
print("row sumsq sample:", np.round((X[:5]**2).sum(axis=1),3))
print("row sum sample:", np.round(X[:5].sum(axis=1),3))
# label correlation quick: top feature corr
y = trf['label'].to_numpy()
corr = np.array([np.corrcoef(X[:,j], y)[0,1] if X[:,j].std()>0 else 0 for j in range(X.shape[1])])
order = np.argsort(-np.abs(corr))[:10]
print("top |corr| feats idx:", order.tolist())
print("their corr:", np.round(corr[order],3).tolist())
