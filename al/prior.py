from torch import FloatTensor


def get_prior_from_model_perspective(probas: FloatTensor) -> FloatTensor:
    return probas.mean(dim=0)
