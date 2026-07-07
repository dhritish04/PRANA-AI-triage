import wfdb

for record in ["100","101","102","103","104"]:

    ann = wfdb.rdann(
        record,
        "atr",
        pn_dir="mitdb"
    )

    print(record)
    print(set(ann.symbol))
    print()