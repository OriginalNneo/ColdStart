from datasets import load_dataset
ds=load_dataset("NicolaiSivesind/ChatGPT-Research-Abstracts", split="train")
print("rows:", len(ds))
import numpy as np
rw=[int(x) for x in ds["real_word_count"]]
gw=[int(x) for x in ds["generated_word_count"]]
print("real wc median/min/max:", int(np.median(rw)), min(rw), max(rw))
print("gen  wc median/min/max:", int(np.median(gw)), min(gw), max(gw))
