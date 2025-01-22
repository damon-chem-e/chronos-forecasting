import datasets

ds = datasets.load_dataset("autogluon/chronos_datasets", "m4_daily", split="train")
ds.set_format("numpy")  # sequences returned as numpy arrays

print(ds)

