# Dataset policy

The authors' GC10-DET release lives locally under `data/gc10-det/`. See the
[dataset record](../docs/dataset.md) for the pinned source, usage terms,
preparation commands, and integrity audit.

Keep local payloads under `data/`, where Git ignores them. Keep only
provenance and split manifests under `data/manifests/`. Do not put images,
annotations, archives, or credentials in that exception.

Before baseline work, record the source URL, access date, release or archive
identity, usage terms, checksums, class mapping, annotation format, split
membership and seed where applicable, and any transformations or exclusions.
Preserve original data and document derived versions separately. Mark unknown
information explicitly instead of guessing it.
