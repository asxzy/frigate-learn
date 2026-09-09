"""Frigate Learn: active-learning / auto-training pipeline for Frigate NVR.

The package is built in phases. Phase 0 (golden dataset + metrics) and Phase 1
(Frigate 0.18 HTTP collector) are implemented; later phases are scaffolded in
their own subpackages.
"""

__version__ = "0.2.0"

from .config import AppConfig, load_config

__all__ = ["__version__", "AppConfig", "load_config"]