from . import (
    RPSstuff,
    colormaps,
    datasets,
    filters,
    ocfis,
    plotting,
    sw_extras,
)


__all__ = [
    RPSstuff,
    colormaps,
    datasets,
    filters,
    ocfis,
    plotting,
    sw_extras,
]


from ._version import get_versions  # noqa
__version__ = get_versions()['version']
del get_versions
