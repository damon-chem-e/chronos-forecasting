f = open("remote_dataset_files.txt", "r")

for line in f:
    print("- \"" + line.strip() + "\"")