# distrubution-extentions

![python](https://img.shields.io/badge/python-3.8-blue)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![lint](https://github.com/nomutin/distribution-extention/actions/workflows/lint.yml/badge.svg)](https://github.com/nomutin/distribution-extention/actions/workflows/lint.yml)
[![test](https://github.com/nomutin/distribution-extention/actions/workflows/test.yml/badge.svg)](https://github.com/nomutin/distribution-extention/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/nomutin/distribution-extention/graph/badge.svg?token=HTHTLULHPV)](https://codecov.io/gh/nomutin/distribution-extention)

Simple `torch.distributions` wrappers for DL.

## API

This module provides the following features.

- Easy Instanitate

    ```python
    import torch
    from distrubution_extentions import NormalFactory

    tensor = torch.rand([256, 100, 16])
    distribution = NormalFactory()(Tensor)
    distribution.sample()  # -> Tensor[256, 100, 8]
    ```

- Easy Independence

    ```python
    distribution = Normal(loc=loc, scale=scale)
    independent = distribution.independent(dim=1)
    ```

- Device Conversion

    ```python
    device = torch.device("cuda:0")
    distribution = Normal(loc=loc, scale=scale)
    distribution = distribution.to(device=device)
    ```

- Slicing

    ```python
    distribution = Normal(loc=loc, scale=scale)[:, 0, :]
    distribution.sample()  # -> Tensor[256, 8]
    ```

- Stop Gradient

    ```python
    distribution = Normal(loc=loc, scale=scale) 
    distribution.detach()
    ```
