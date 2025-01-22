import datasets

path = "weekly"
ds = datasets.load_dataset("autogluon/chronos_datasets", path, split="train")

print(ds)