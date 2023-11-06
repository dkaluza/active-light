import torch


def uniform_mesh(n_classes, step=0.01):
    proba_range = torch.range(0, 1 + step, step=step)
    distribution = torch.meshgrid(
        *[proba_range for _ in range(n_classes)], indexing="ij"
    )
    distribution = torch.stack(distribution, dim=1)

    distribution = distribution[torch.sum(distribution, dim=1) <= 1]
    return distribution
