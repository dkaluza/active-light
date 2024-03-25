from torch import FloatTensor


def get_prior_from_model_perspective(
    probas: FloatTensor, keepdim: bool = False
) -> FloatTensor:
    return probas.mean(dim=0, keepdim=keepdim)
