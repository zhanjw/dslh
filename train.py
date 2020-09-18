import time

import torch.nn.functional as F

from dataset import *
from utils import *

torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.set_device(0)


def train_val(config, bit):
    if config['GPU']:
        device = "cuda"
    train_loader, test_loader, dataset_loader, num_train, num_test = get_data(config)
    class_set, class_loader, class_num, loader_num = get_training_data_of_certain_class(config)
    net = config["net"](bit).to(device)

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    PROTOTYPES = {}
    HASH = torch.zeros(num_train, bit).float().to(device)
    FEAT = torch.zeros(num_train, 256 * 6 * 6).float().to(device)
    LABL = torch.zeros(num_train, config["n_class"]).float().to(device)
    SIGM = torch.zeros(config["n_class"], num_train).float().to(device)
    CORR = torch.zeros(num_train, config["n_class"]).float().to(device)
    Best_mAP = 0
    for epoch in range(config["epoch"]):
        # net.scale = (epoch // config["step_continuation"] + 1) ** 0.5
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")
        net.train()
        h1s_loss = 0
        h2s_loss = 0
        quan_loss = 0
        corr_loss = 0
        train_loss = 0

        # TRAINING PHASE
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            f, b = net(image)
            r = torch.tanh(b)

            HASH[ind, :] = r.data
            FEAT[ind, :] = f.data
            LABL[ind, :] = label.float()

            s1, s1s = get_soft_similarity_matrix(LABL[ind, :], LABL)
            s2, s2s = get_soft_similarity_matrix(CORR[ind, :], CORR)

            hash_loss1s = ((bit * s1s - r @ HASH.t()) ** 2).sum()
            hash_loss2s = ((bit * s2s - r @ HASH.t()) ** 2).sum()

            quantization_loss = ((b - b.sign()) ** 2).sum()
            correlation_loss = (b @ torch.ones(b.shape[1], 1, device=b.device)).sum()

            hash_loss1s = hash_loss1s / (b.shape[0] * HASH.shape[0]) / 2
            hash_loss2s = hash_loss2s / (b.shape[0] * HASH.shape[0]) / 2
            quantization_loss = quantization_loss * 200 / (b.shape[0] * HASH.shape[0])
            correlation_loss = correlation_loss * 50 / (b.shape[0] * HASH.shape[0])
            loss = (1 - config["s2_scale"]) * hash_loss1s + config[
                "s2_scale"] * hash_loss2s + quantization_loss + correlation_loss

            h1s_loss += hash_loss1s
            h2s_loss += hash_loss2s
            quan_loss += quantization_loss
            corr_loss += correlation_loss
            train_loss += loss

            if epoch < config["start_epoch"]:
                ls = loss
            else:
                ls = loss
            ls.backward()
            optimizer.step()

        print("\b\b\b\b\b\b\b loss:%.3f, h1:%.3f, h2:%.3f, quan:%.3f, corr:%.3f" % (
            train_loss, h1s_loss, h2s_loss, quan_loss, corr_loss))

        # CORRECTION PHASE
        if epoch >= config["start_epoch"] - 1:
            net.eval()
            FEAT_normalize = F.normalize(FEAT)
            for idx, class_key in enumerate(class_loader):
                for image, label, ind in class_loader[class_key]:
                    image = image.to(device)
                    
                    with torch.no_grad():
                        f, b = net(image)

                    f_normalize = F.normalize(f, dim=-1)
                    s_ij = f_normalize.mm(f_normalize.t())
                    s_ranked, s_ranked_idx = torch.sort(s_ij)
                    s_const = s_ranked[:, int(s_ranked.shape[1] * 0.5)]
                    min_s_for_max, _ = torch.min(s_ij, 1)

                    rho = torch.sum(torch.sign(s_ij - s_const), 1)
                    rho_j_gt_rho_i = (rho.unsqueeze(1) < rho.unsqueeze(0))
                    max_s_for_most, _ = torch.max(rho_j_gt_rho_i.float() * s_ij, 1)

                    eta = torch.where(rho < torch.max(rho), max_s_for_most, min_s_for_max)
                    eta_threshold = torch.min(eta) + config["eta_threshold"] * (torch.max(eta) - torch.min(eta))

                    select_eta_constraint = torch.where(eta < eta_threshold,
                                                        torch.ones_like(eta).to(device),
                                                        torch.zeros_like(eta).to(device))

                    density_threshold, _ = torch.sort(rho * select_eta_constraint, descending=True)

                    density_threshold = density_threshold[config["density_threshold"]]

                    select_density_constraint = torch.where(rho > density_threshold,
                                                            select_eta_constraint,
                                                            torch.zeros_like(eta).to(device))
                    PROTOTYPES[class_key] = select_density_constraint.long()
                    select = (PROTOTYPES[class_key] == 1).nonzero().squeeze()
                    f_selected = f[select]
                    f_selected_normalize = F.normalize(f_selected, dim=-1)
                    if len(f_selected_normalize.shape) < 2:
                        f_selected_normalize = torch.unsqueeze(f_selected_normalize, 0)
                    cosine_distance = FEAT_normalize.mm(f_selected_normalize.t())
                    cosine_distance = torch.mean(cosine_distance, 1)
                    SIGM[idx] = cosine_distance

            corrected_label = torch.argmax(SIGM, dim=0).unsqueeze(1)
            CORR.zero_()
            CORR.scatter_(1, corrected_label, 1)

        if (epoch + 1) % config["evaluate_freq"] == 0:
            print("calculating test binary code......")
            tst_binary, tst_label = compute_result(test_loader, net, usegpu=config["GPU"])

            print("calculating dataset binary code.......")
            # trn_binary, trn_label = compute_result(train_loader, net, usegpu=config["GPU"])
            trn_binary, trn_label = compute_result(dataset_loader, net, usegpu=config["GPU"])

            print("calculating map.......")
            mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                             config["topK"])
            print(
                "%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f" % (
                    config["info"], epoch + 1, bit, config["dataset"], mAP))
            print(config)
            if mAP > Best_mAP:
                Best_mAP = mAP
    print("bit:%d,Best MAP:%.3f" % (bit, Best_mAP))
