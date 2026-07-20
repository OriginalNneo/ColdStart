from datasets import load_dataset
def probe(name, cfg=None):
    try:
        if cfg:
            ds = load_dataset(name, cfg, split="train", streaming=True)
        else:
            ds = load_dataset(name, split="train", streaming=True)
        it = iter(ds); rows=[next(it) for _ in range(3)]
        print(f"OK {name} cfg={cfg}: keys={list(rows[0].keys())}")
        for r in rows:
            print("   ", {k:(str(v)[:70]) for k,v in r.items()})
    except Exception as e:
        print(f"FAIL {name} cfg={cfg}: {type(e).__name__}: {str(e)[:150]}")
for n in ["NicolaiSivesind/human-vs-machine","andythetechnerd03/AI-human-text","Hello-SimpleAI/HC3"]:
    probe(n)
