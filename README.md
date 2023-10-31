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
    distribution = Normal(loc=loc, scale=scale)
    independent = distribution.independent(dim=1)
    ```

- Device conversion

    ```python
    device = torch.device("cuda:0")
    distribution = Normal(loc=loc, scale=scale)
    distribution = distribution.to(device=device)
    ```

- Slicing

    ```python
    distribution = Normal(loc=loc, scale=scale) [:, 0, :]
    distribution.sample()  # -> Tensor[128, 8]
    ```

- Gradient stop

    ```python
    distribution = Normal(loc=loc, scale=scale) 
    distribution.detach()
    ```
