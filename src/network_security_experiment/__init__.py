"""Shared-storage covert-channel measurement experiment.

Two LLM task-agents (A, B) collaborate on SILO-BENCH distributed tasks and may
exchange information only through three shared-storage carriers:

- M1: file *content* (fixed framework-assigned paths, payload in file body)
- M2: directory *names* (base32-encoded payload in empty-dir names)
- M3: shared-cache *key existence* signaling (1 bit per key, DiskCache backend)

The measurement layer sits between agent tools and the storage backends and
records per-chunk ground truth, decode candidates, timing, storage operations
and lifecycle events.  Seven channel metrics are computed offline (§8 of the
design doc); agent task answers are used only for run sanity checks, never for
scoring.
"""

__version__ = "0.1.0"