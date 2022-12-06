from typing import Dict


# based on the code of BasicSR, mmcv
class Registry:

    def __init__(self, name):
        super(Registry, self).__init__()
        self._name = name
        self._records = {}

    def register(self, name=None):
        def _register(cls_or_func):
            registry_name = cls_or_func.__name__ if name is None else name
            if registry_name in self._records:
                raise ValueError(f'Target {registry_name} already in Registry({self._name})')
            self._records[registry_name] = cls_or_func

        return _register

    def get(self, registry_name):
        """Get the registry record.
        Args:
            registry_name (str): The registry name in string format.
        Returns:
            class_or_function: The corresponding class or function.
       """
        if registry_name not in self._records:
            raise KeyError(f'No registry named {registry_name} found in Registry({self._name})')
        return self._records[registry_name]

    def __len__(self):
        return len(self._records)

    def __contains__(self, item):
        return item in self._records


NETWORK_REGISTRY = Registry('network')
