# Dataset policy

GC10-DET is the intended primary dataset. Acquisition source, version, usage
terms, and split protocol have not yet been verified for this repository.
No dataset has been downloaded during foundation setup.

Place future local payloads under `data/`, where Git ignores them. Keep only
small provenance and split manifests under `data/manifests/` when available;
do not put images, annotations, archives, or credentials in that exception.

Before baseline work, record the source URL, access date, release or archive
identity, usage terms, checksums, class mapping, annotation format, split
membership and seed where applicable, and any transformations or exclusions.
Preserve original data and document derived versions separately. Mark unknown
information explicitly instead of guessing it.
