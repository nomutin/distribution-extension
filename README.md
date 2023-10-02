# distrubution-extentions

Simple `torch.distributions` wrappers for DL.

## API

This module provides the following features.

- Easy instanitate

    ```python
    import torch
    from distrubution_extentions import NormalFactory

    tensor = torch.rand([256, 100, 16])
    distribution = NormalFactory()(Tensor)
    distribution.sample()  # -> Tensor[128, 100, 8]
    ```

- Easy independence

    ```python
    distribution = Normal.from_tensor(tensor=tensor)
    independent = distribution.independent(dim=-1)
    ```

- Device conversion

    ```python
    device = torch.device("cuda:0")
    distribution = Normal.from_tensor(tensor=tensor)
    distribution = distribution.to(device=device)
    ```

- Slicing

    ```python
    distribution = Normal.from_tensor(tensor=tensor)[:, 0, :]
    distribution.sample()  # -> Tensor[128, 8]
    ```

- Gradient stop

    ```python
    distribution = Normal.from_tensor(tensor=tensor)
    distribution.detach()
    ```
