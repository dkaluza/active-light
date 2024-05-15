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

GPU version currently requires in conda installation and CUDA capable GPU.

If you would like to use gpu version of gpu capable algorithms then install appropriate
requirements with:

```
poe install-gpu
```

> [!NOTE]
> If installation on Windows leads to an error, e.g. `'chcp' is not recognized as an internal or external command`
> Or while importing library you encounter an error `ImportError: DLL load failed while importing _swigfaiss: The specified module could not be found.`
> Then try to uninstall faiss-gpu and faiss-cpu and run the command in anaconda prompt.