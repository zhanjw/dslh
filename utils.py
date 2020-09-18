import torch

def get_soft_similarity_matrix(train_targets, seen_targets):
    S = (train_targets @ seen_targets.t() > 0).float()
    S = torch.where(S == 1, torch.full_like(S, 1), torch.full_like(S, -1))

    # Soft similarity matrix, benefit to converge
    r = S.sum() / (1 - S).sum()
    SS = S * (1 + r) - r
    return S, SS