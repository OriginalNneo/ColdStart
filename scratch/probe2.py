from datasets import load_dataset, get_dataset_config_names
n="NicolaiSivesind/ChatGPT-Research-Abstracts"
try:
    print("configs:", get_dataset_config_names(n))
except Exception as e:
    print("cfgfail", e)
try:
    ds=load_dataset(n, split="train", streaming=True)
    it=iter(ds); rows=[next(it) for _ in range(4)]
    print("keys:",list(rows[0].keys()))
    for r in rows:
        print("  ",{k:(str(v)[:90]) for k,v in r.items()})
except Exception as e:
    print("loadfail",type(e).__name__,str(e)[:200])
