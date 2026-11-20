# GC10-DET dataset

FlawTrace uses the preprocessed GC10-DET splits supplied by the AGE-YOLO
authors. The [pinned Hugging Face revision](https://huggingface.co/datasets/YYYHHHH993/AGE-YOLO/tree/93d2bc4420b024d3cbd825610ad7aad7b2544311)
is `93d2bc4420b024d3cbd825610ad7aad7b2544311`, accessed November 20, 2026.
The dataset card declares **CC BY-NC 4.0**, which requires attribution and
limits licensed use to noncommercial purposes. The card gives no separate
GC10-specific terms. [Creative Commons license](https://creativecommons.org/licenses/by-nc/4.0/)

## Archive and acquisition

The published RAR contains separate `gc10/`, `pcb/`, and `neu/` trees.
The RAR5 Quick Open index and adjacent member headers place all GC10 members
in the contiguous prefix at bytes 0–961,556,759 inclusive. Preparation
downloads that exact range, requires HTTP 206 with the expected
`Content-Range`, and verifies the prefix SHA-256. It appends the original
eight-byte RAR end marker to a temporary extraction copy. The extracted
GC10 files retain their original bytes and split membership.

| Artifact | Size in bytes | SHA-256 |
| --- | ---: | --- |
| Retained GC10 download | 961,556,760 | `9d9370dceb77ceb05de7d3c571ce89632fb9430b83b0f47c86058794295a3b18` |
| Original combined RAR | 1,882,507,969 | `fb6b6b9d9709a75b05f0adde5315209a0d03b2e182fc428c181c5aab72bd5b13` (published LFS checksum) |
| Extracted GC10 tree | 961,231,330 | Per-file checksums in the split and provenance manifests. |

The recorded GC10 download is 917.0 MiB. The extracted size is 916.7 MiB
across 4,590 files. The combined RAR checksum is published by Hugging Face.
The prefix checksum was verified locally.

```text
data/
├── .downloads/gc10-prefix.bin             # ignored
├── gc10-det/                             # ignored
│   ├── gc10_data.yaml                     # author's original
│   ├── testlist_gc10.txt
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/
└── manifests/
    ├── gc10-provenance.json
    ├── gc10-audit.json
    ├── gc10-train.csv
    ├── gc10-val.csv
    └── gc10-test.csv
```

## Preparation and validation

Initialize the AGE-YOLO submodule and follow the
[Python environment setup](../README.md#environment-installation). Install
`libarchive-tools` if `bsdtar` is missing on Ubuntu. From the repository root:

```sh
sudo apt-get install libarchive-tools  # only if bsdtar is missing
python scripts/prepare_data.py
python scripts/check_data.py
```

Preparation reuses a verified cached prefix if present and refuses to
overwrite an existing dataset tree. Allow about 4 GiB of free disk space for
the download, temporary extraction files, and final tree. Validation is
read-only and reports whether the observed paths, checksums, and audit facts
match the recorded manifests. It can pass with known findings if the data
still match the record.

The [FlawTrace YAML](../configs/gc10.yaml) uses the verified ID order below.
Pass it by absolute filename to the pinned AGE-YOLO loader so it resolves the
split paths relative to the YAML directory.

## Split and class counts

All 2,294 images decoded at **2048 × 1000** pixels. The archived test list
matches all 230 test image stems. Objects are counted as nonblank YOLO
annotation rows.

| Split | Images | Label files | Objects | Empty label files |
| --- | ---: | ---: | ---: | ---: |
| Train | 1,834 | 1,834 | 2,881 | 2 |
| Validation | 230 | 230 | 345 | 0 |
| Test | 230 | 230 | 337 | 0 |
| **Total** | **2,294** | **2,294** | **3,563** | **2** |

| ID | Class | Train | Validation | Test | Total |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | `punching_hole` | 283 | 24 | 22 | 329 |
| 1 | `welding_line` | 425 | 46 | 42 | 513 |
| 2 | `crescent_gap` | 219 | 26 | 20 | 265 |
| 3 | `water_spot` | 293 | 35 | 26 | 354 |
| 4 | `oil_spot` | 460 | 57 | 52 | 569 |
| 5 | `silk_spot` | 713 | 76 | 95 | 884 |
| 6 | `inclusion` | 263 | 48 | 36 | 347 |
| 7 | `rolled_pit` | 75 | 4 | 6 | 85 |
| 8 | `crease` | 43 | 19 | 12 | 74 |
| 9 | `waist_folding` | 107 | 10 | 26 | 143 |

## Integrity and open questions

- The archive YAML, pinned AGE-YOLO YAML, and FlawTrace YAML agree on all ten
  class IDs. Every image has one matching label file in its split. No orphan
  labels, malformed rows, invalid class IDs, out-of-bounds boxes, or decoding
  failures were found. Box edges were checked with a normalized tolerance of
  `1e-6`. No edges needed that tolerance.
- Two training label files contain no objects:
  `img_02_4406772100_00175.txt` and `img_06_3402617700_00988.txt`.
  Both were retained. Whether these are intended negatives or missing
  annotations remains unresolved.
- No exact duplicates were found within or across any split, either by source
  file SHA-256 or by decoded RGB pixels and dimensions. Near duplicates,
  related production samples, and annotation correctness were outside this
  audit. Split membership remains the authors' original.

This audit covers the original release. B1 offline augmentation is recorded in [the baseline method](baseline.md). Full findings and paths are recorded in
[the audit manifest](../data/manifests/gc10-audit.json).
