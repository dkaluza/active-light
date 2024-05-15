# Active Light
A library for active learning knowledge aquisition based on pytorch.

## Installation

### 1. First make sure you have poetry installed.
E.g.:
```
pipx install poetry
```


### 2. Next install the package:

```
poetry install
```


### 3. Install extras using poe.

Some of the submodules e.g. clustering and loop experiments need additional optional dependencies.
You can install them with the following command:
```
poe install-all
```

### 4. GPU version.

If you would like to use gpu version of gpu capable algorithms then install appropriate
requirements with:

```
poe install-gpu
```