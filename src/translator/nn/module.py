import torch.nn


class Module(torch.nn.Module):
    @property
    def device(self) -> torch.device:
        """Returns the module's device.

        This property requires all submodules to be on the same device.

        Returns:
            The module's device.
        """
        return next(self.parameters()).device
