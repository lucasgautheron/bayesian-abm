# Processed MemeTracker story data

These Parquet files are derived from `../clust-qt08080902w3mfq5.txt.gz`,
the [MemeTracker phrase-cluster dataset](https://snap.stanford.edu/memetracker/data.html).
In this directory, a **story** means a MemeTracker cluster of related quoted
phrases. It does not mean an individual article; articles and blog posts are
represented by URLs in the source data.

The original gzip is the lossless, event-level archive. It is intentionally
not duplicated as an event-level Parquet file.

## Files

### `stories.parquet`

One row per story (71,568 rows).

| Column | Type | Meaning |
| --- | --- | --- |
| `story_id` | int64 | Source cluster ID |
| `representative_phrase` | string | Source cluster's root phrase |
| `phrase_count` | int32 | Number of phrase variants in the cluster |
| `total_mentions` | int64 | Mentions across all variants and URLs |

### `phrases.parquet`

One row per phrase variant (310,457 rows).

| Column | Type | Meaning |
| --- | --- | --- |
| `phrase_id` | int64 | Source phrase ID |
| `story_id` | int64 | Story containing this phrase |
| `phrase` | string | Phrase text |
| `total_mentions` | int64 | Mentions of the phrase across all URLs |
| `url_count` | int32 | Source URL-observation count for the phrase |

### `story_daily.parquet`

A dense daily panel with one row for every story on every date from
2008-08-01 through 2009-01-31, inclusive (184 rows per story and 13,168,512
rows total). Inactive days are present with integer zeros; count columns are
never null.

The analysis loader reads the complete population as one joint observation.
The shared story-summary pipeline ranks stories by total mentions, retains
the top 1,000, and pads smaller populations with an explicit validity mask.
A shared temporal encoder and DeepSet learn fixed-size inference conditions
from this panel. Simulations use the same ranking and truncation computation.

| Column | Type | Meaning |
| --- | --- | --- |
| `story_id` | int64 | Story ID |
| `date` | date32 | Publication date |
| `mentions` | int64 | Sum of source `Fq` across the story's URL observations |
| `blog_mentions` | int64 | Mentions at blog URLs (`UrlTy=B`) |
| `mainstream_mentions` | int64 | Mentions at mainstream-media URLs (`UrlTy=M`) |
| `urls` | int32 | Distinct URLs across all phrase variants |
| `blog_urls` | int32 | Distinct blog URLs |
| `mainstream_urls` | int32 | Distinct mainstream-media URLs |
| `active_phrases` | int32 | Distinct phrase variants observed that day |

`mentions` equals `blog_mentions + mainstream_mentions`. URL columns count
distinct URL strings, so repeated source observations of phrase variants at
the same URL do not inflate the daily URL count.

## Rebuild

Requires Python and PyArrow:

```sh
python3 scripts/convert_memetracker.py \
  data/clust-qt08080902w3mfq5.txt.gz \
  data/processed
```

The converter streams one story at a time, validates the source hierarchy and
declared totals, and atomically replaces completed Parquet outputs. Generated
files use Zstandard compression.
