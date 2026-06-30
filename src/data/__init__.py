"""Data processing module.

Some utilities (e.g., sampling / manifest tooling) do not require heavy ML
dependencies like torch. Keep package import resilient so those scripts can run
in lightweight environments.
"""

try:
	from .data_processing import *  # type: ignore
except Exception:
	# Optional dependency (e.g., torch) may be missing in evaluation-only envs.
	pass