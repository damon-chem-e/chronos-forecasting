import datasets

path = "m4_daily"
ds = datasets.load_dataset("autogluon/chronos_datasets", path, split="train")

print(ds.features)