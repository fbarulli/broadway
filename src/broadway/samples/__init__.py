"""Named-sample registry: definition → immutable artifact → validated consumption.

Pipeline steps declare a sample name; the registry resolves it to an
immutable versioned artifact and the loader validates provenance, integrity,
row count, and schema before returning it.
"""

from __future__ import annotations

from broadway.samples.generate import generate_sample
from broadway.samples.loader import read_named_sample
from broadway.samples.models import Sample

__all__ = ["Sample", "generate_sample", "read_named_sample"]
