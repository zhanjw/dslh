import numpy as np
import os

data_path = "data"
# dataset = "coco"
for dataset in os.listdir(data_path):
    print(dataset)
    image_list = open(data_path + "\\" + dataset + "\\train.txt")
    imgs = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
    for j in range(len(imgs)):
        path, target = imgs[j]
        for i in range(len(target)):
            if target[i]:
                f = open(data_path + "\\" + dataset + "\\class" + str(i + 1) + ".txt", "a")
                t = ' '.join(str(i) for i in target)
                f.write(path + ' ' + t + '\n')
                f.close()
