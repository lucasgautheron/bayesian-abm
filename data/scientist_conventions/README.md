# Scientist convention networks

These files reproduce the scientist networks used in
[Dilemmas and trade-offs in the diffusion of conventions](https://arxiv.org/abs/2501.17300).
They are derived from the paper's
[public GIN repository](https://gin.g-node.org/lucasgautheron/dilemmas-conventions)
and its INSPIRE DataLad subdataset.

The generated Parquet files are ignored by Git, like the project's other
processed datasets.

The current conversion contains 42,884 scientists, including 2,475 with an
observed favorite convention, 1,311,572 coauthorship edges, and 213,048
directed citation edges.

## Files

### `scientists.parquet`

One row per scientist in the convention-labelled publication population.

| Column | Type | Meaning |
| --- | --- | --- |
| `scientist_id` | int32 | Stable dense ID, ordered by source author ID |
| `source_author_id` | string | INSPIRE BAI author identifier |
| `favorite_convention` | nullable int8 | `-1` mostly-minus, `+1` mostly-plus, null if unknown |
| `primary_area` | int8 | Index into the research-area list below |
| `career_start_year` | int16 | First dated publication year |
| `area_share_0` ... `area_share_3` | float32 | Fraction of publications assigned to each area |

The cultural-transmission models use 1981 as their random-walk baseline, as in
the paper, and map earlier career starts to that baseline year.
Area labels are not mutually exclusive, so the four publication fractions need
not sum to one.

Research-area indices are:

0. Phenomenology-HEP
1. Theory-HEP
2. Gravitation and Cosmology
3. Astrophysics

### `coauthorship.parquet`

An undirected weighted edge list. Endpoints use `scientist_id`, are canonical
(`source < target`), and occur once.

| Column | Type | Meaning |
| --- | --- | --- |
| `source` | int32 | First scientist |
| `target` | int32 | Second scientist |
| `weight` | float64 | Sum of `1 / (authors on paper - 1)` |
| `first_year` | int16 | First shared publication year |

As in the paper, publications with more than 100 authors are excluded.

### `citations.parquet`

A directed weighted edge list between scientists with observed favorite
conventions.

| Column | Type | Meaning |
| --- | --- | --- |
| `source` | int32 | Scientist on the citing paper |
| `target` | int32 | Scientist on the cited paper |
| `weight` | float64 | Sum of `1 / (citing authors * cited authors)` |

Self-citations at the scientist level are omitted.

## Retrieve and rebuild

Install DataLad, then retrieve the source:

```sh
datalad install -r \
  -s https://gin.g-node.org/lucasgautheron/dilemmas-conventions.git \
  data/scientist_conventions/source
datalad get \
  data/scientist_conventions/source/data/authors_signatures.parquet
datalad get \
  data/scientist_conventions/source/inspire-harvest/database/articles.parquet \
  data/scientist_conventions/source/inspire-harvest/database/articles_authors.parquet \
  data/scientist_conventions/source/inspire-harvest/database/articles_references.parquet
```

The GIN HTTPS remote does not advertise the annexed `signature.parquet` to
git-annex. Retrieve that published file directly:

```sh
rm data/scientist_conventions/source/data/signature.parquet
curl -fL \
  https://gin.g-node.org/lucasgautheron/dilemmas-conventions/raw/master/data/signature.parquet \
  -o data/scientist_conventions/source/data/signature.parquet
```

Convert it:

```sh
python3 data/convert_scientist_conventions.py \
  data/scientist_conventions/source \
  data/scientist_conventions
```

The converter validates required columns, convention values, dated research
areas, edge normalization, and source availability. It writes
Zstandard-compressed files atomically.
