import torch


def uniform_mesh(n_classes, step=0.01):
    proba_range = torch.arange(0, 1 + step, step=step)

    # we ommit a last proba value as it is uniquely determined by combination of others
    distribution = torch.cartesian_prod(*[proba_range for _ in range(n_classes - 1)])

    distribution = distribution[torch.sum(distribution, dim=1) <= 1]
    distribution = torch.concat(
        (distribution, 1 - distribution.sum(dim=1).unsqueeze(dim=1)), dim=1
    )
    return distribution
