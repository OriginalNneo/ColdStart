import pandas as pd, numpy as np
for f in ["data/train.csv","data/test.csv","data/sample_submission.csv"]:
    df = pd.read_csv(f, nrows=5)
    print(f, "->", list(df.columns))
tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
print("train rows", len(tr), "label dist", tr['label'].value_counts().to_dict())
print("test rows", len(te))
print("test id sample:", te['id'].head(3).tolist())
print("train text len stats:", tr['text'].str.len().describe()[['mean','min','max']].round(1).to_dict())
# features files
trf = pd.read_csv("data/train_features.csv", nrows=3)
print("train_features cols n=", trf.shape[1], "first:", list(trf.columns)[:6], "... last:", list(trf.columns)[-3:])
print("train_features dtypes sample:", trf.dtypes.iloc[1:4].tolist())
tef = pd.read_csv("data/test_features.csv", nrows=3)
print("test_features cols n=", tef.shape[1], "first:", list(tef.columns)[:6])
