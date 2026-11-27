"""Team harness ACE hook. Importing this module never imports a model client."""
from .store import on_batch

__all__ = ['on_batch']
