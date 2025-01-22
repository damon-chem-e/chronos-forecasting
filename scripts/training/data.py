import datasets
import numpy as np

ds = datasets.load_dataset("autogluon/chronos_datasets", "mexico_city_bikes", split="train")
ds.set_format("numpy")  # sequences returned as numpy arrays

new_col = [np.datetime64("2000-01-01 00:00", "s")] * len(ds)
ds = ds.add_column("start", new_col)

print(ds)

