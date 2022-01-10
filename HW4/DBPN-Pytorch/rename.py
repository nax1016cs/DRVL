import os
path = "./result/"
entries = os.listdir(path)
for filename in entries:
    id_ = filename.split(".")[0]
    os.rename(path + filename, path+id_ + "_pred.png")
