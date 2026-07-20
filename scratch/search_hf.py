from huggingface_hub import list_datasets
qs=["machine generated scientific abstract","AI generated abstract detection","GPT abstract human","LLM scientific text detection","chatgpt abstracts"]
seen=set()
for q in qs:
    try:
        for d in list(list_datasets(search=q, limit=12)):
            if d.id in seen: continue
            seen.add(d.id)
            dl=getattr(d,'downloads',None)
            print(f"{d.id}  dl={dl}")
    except Exception as e:
        print("qfail",q,e)
