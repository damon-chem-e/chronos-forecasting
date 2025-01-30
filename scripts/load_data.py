from datasets import load_dataset, get_dataset_config_names

# gets all subset names
configs = get_dataset_config_names("autogluon/chronos_datasets") 
# print(configs)
exclude = ["training_corpus_kernel_synth_1m", "training_corpus_tsmixup_10m"]

REMOTE_PATH = "/nfs/sloanlab007/projects/chimera_proj/chronos_datasets/hugging_face_datasets/data"

for config in configs[:2]:
    ds = load_dataset("autogluon/chronos_datasets", config, split="train")
    arrow_files = [file["filename"] for file in ds.cache_files]
    print(arrow_files)


