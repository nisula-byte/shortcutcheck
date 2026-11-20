"""Metamorphic relations: ``base`` (interface), ``core`` (MR1-MR4), ``secondary`` (MR5-MR6)."""

from shortcutcheck.mrs.base import (
    MetamorphicRelation,
    NotApplicableError,
    NotFittedError,
    ObservedRange,
    changed_columns,
    changed_rows,
)
from shortcutcheck.mrs.core import MetadataSwap, SdkShift, SizeScale, YearShift, core_mrs
from shortcutcheck.mrs.realism import RealismCheck
from shortcutcheck.mrs.secondary import AddDangerousPermission, AddNormalPermission, secondary_mrs

__all__ = [
    "AddDangerousPermission",
    "AddNormalPermission",
    "MetadataSwap",
    "MetamorphicRelation",
    "NotApplicableError",
    "NotFittedError",
    "ObservedRange",
    "RealismCheck",
    "SdkShift",
    "SizeScale",
    "YearShift",
    "changed_columns",
    "changed_rows",
    "core_mrs",
    "secondary_mrs",
]
