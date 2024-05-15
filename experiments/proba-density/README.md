# Fast Approximate Loss Change for active learning on imbalanced data


This branch contains experiments and code described in publication
"Fast Approximate Loss Change for active learning on imbalanced data"
written by Daniel Kaluza and Andrzej Janusz.

Source code of the proposed FALC method is splitted into 2 classes:
* Implementation of density estimation in prediction space is available in [PrDensity](/al/sampling/repr/pr_density.py#L17).
* Aproximate weighted entropy loss change is available in [ClassWeightedEntropy](/al/sampling/uncert/classification/prior.py#L60).

Utitlity values from those 2 classes are aggregated with multiplication as described in the paper.

The code to reproduce the experiments is available [here](classification_experiments.ipynb).

## Installation

To install all prerequisits please follow the main [README](/README.md) of the repository, to be able to reproduce the experiments please install all of the dependency groups and extras.

## Result Highlights

Proposed FALC method has been tested against references from the literature on 6 fully annotated data sets commonly used in machine learning research. The datasets had majority of their annotations hidden to simulate the active learning scenario.

### Ranks
Average rank of FALC methods acrros the datasets (the lower the better):
| Dataset      | $FALC^{sc}_u$ | $FALC^{si}_u$ | $FALC^{si}_g$ | $FALC^{sc}_g$ | Random |
|--------------|---------------|---------------|---------------|---------------|--------|
| Mnist        | 3             | 1             | 4             | 2             | 8      |
| Firefighters | 1             | 2             | 3             | 4             | 9      |
| Covertype    | 3             | 1             | 2             | 5             | 7      |
| Landsat      | 2             | 3             | 4             | 1             | 9      |
| Letter       | 3             | 6             | 4             | 5             | 8      |
| Cybersec     | 4             | 2             | 9             | 1             | 6      |
| Average      | 2.67          | 2.5           | 4.33          | 3.0           | 7.83   |

Reference methods:
| Dataset      | KNN-Entropy | BootstrapJS | Random | Entropy | Off-Centered |
|--------------|-------------|-------------|--------|---------|--------------|
| Mnist        | 9           | 7           | 8      | 5       | 6            |
| Firefighters | 8           | 6           | 9      | 5       | 7            |
| Covertype    | 9           | 8           | 7      | 4       | 6            |
| Landsat      | 8           | 6           | 9      | 5       | 7            |
| Letter       | 1           | 9           | 8      | 2       | 7            |
| Cybersec     | 7           | 3           | 6      | 8       | 5            |
| Average      | 7.0         | 6.5         | 7.83   | 4.83    | 6.33         |


### BAC across iterations

Mnist:

![BAC after thresholding on Mnist dataset][mnist_bac]

[mnist_bac]: images/BAC_from_predict_mnist_final.png "BAC across iterations after thresholding on Mnist dataset"

Firefighters:

![BAC after thresholding on Firefighters dataset][firefighters_bac]

[firefighters_bac]: images/BAC_from_predict_firefighters_final.png "BAC across iterations after thresholding on Firefighters dataset"

Covertype:

![BAC after thresholding on Covertype dataset][covertype_bac]

[covertype_bac]: images/BAC_from_predict_covertype_final.png "BAC across iterations after thresholding on Covertype dataset"

Landsat:

![BAC after thresholding on Landsat dataset][landsat_bac]

[landsat_bac]: images/BAC_from_predict_satimage_final.png "BAC across iterations after thresholding on Landsat dataset"

Letter:

![BAC after thresholding on Letter dataset][letter_bac]

[letter_bac]: images/BAC_from_predict_letter_final.png "BAC across iterations after thresholding on Letter dataset"


Cybersec:

![BAC after thresholding on Cybersec dataset][cybersec_bac]

[cybersec_bac]: images/BAC_from_predict_sod_final.png "BAC across iterations after thresholding on Cybersec dataset"