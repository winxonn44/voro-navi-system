"""voronav's optional C++ core. Install with `pip install -e ./core` from the
repository root; the main package selects it with `backend="cpp"`."""

from ._core import data_address, fill_flat_index, roundtrip, version

__all__ = ["data_address", "fill_flat_index", "roundtrip", "version"]
__version__ = version()
