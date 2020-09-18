import numpy as np
import torch
import torch.utils.data as util_data
from PIL import Image
from torchvision import transforms


class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        # print(self.imgs)
        # print(self.imgs[index])
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)


def image_transform(resize_size, crop_size, data_set):
    if data_set == "train_set":
        step = [transforms.RandomHorizontalFlip()]
    else:
        step = []
    return transforms.Compose([transforms.Resize(resize_size),
                               transforms.CenterCrop(crop_size)]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])


def get_data(config):
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    for data_set in ["train_set", "test", "database"]:
        dsets[data_set] = ImageList(config["data_path"],
                                    open(data_config[data_set]["list_path"]).readlines(), \
                                    transform=image_transform(config["resize_size"], config["crop_size"], data_set))
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set], \
                                                      batch_size=data_config[data_set]["batch_size"], \
                                                      shuffle=True, num_workers=4)

    return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], len(dsets["train_set"]), len(
        dsets["test"])


def get_training_data_of_certain_class(config):
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    # for data_set in ["train_set"]:
    for i in range(config["n_class"]):
        data_set = "class" + str(i + 1)
        dsets[data_set] = ImageList(config["data_path"],
                                    open(data_config[data_set]["list_path"]).readlines(), \
                                    transform=image_transform(config["resize_size"], config["crop_size"], data_set))
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set], \
                                                      batch_size=data_config[data_set]["batch_size"], \
                                                      shuffle=True, num_workers=4)
        # dset_loaders[data_set] = util_data.DataLoader(dsets[data_set], \
        #                                               batch_size=len(dsets[data_set]), \
        #                                               shuffle=False, num_workers=4)

    return dsets, dset_loaders, len(dsets), len(dset_loaders)


def compute_result(dataloader, net, usegpu=False):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in dataloader:
        clses.append(cls)
        if usegpu:
            with torch.no_grad():
                f, b = net(img.cuda())
            bs.append(b.data.cpu())
        else:
            with torch.no_grad():
                f, b = net(img)
            bs.append(b.data.cpu())
    return torch.sign(torch.cat(bs)), torch.cat(clses)


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap
