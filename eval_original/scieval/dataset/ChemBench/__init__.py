"""ChemBench dataset entry-point for VLMEvalKit."""
#* Package init for the ChemBench adapter.
#* Keeps imports lightweight and disables ChemBench-specific logging by default.
from loguru import logger

#* Avoid noisy logs during benchmark runs unless explicitly enabled by the caller.
logger.disable("chembench")

#* Adapter version (local to this integration).
__version__ = "0.3.0"

#* Re-export the dataset class so callers can do `from ...ChemBench import ChemBench`.
from .chembench import ChemBench

__all__ = ["ChemBench", "__version__"]
