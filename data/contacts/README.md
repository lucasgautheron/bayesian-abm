# Contact data

`contacts.parquet` is an analysis-ready conversion of `original.dat`. Each row
represents one pairwise contact active during a 20-second interval.

## Schema

| Column | Type | Meaning |
| --- | --- | --- |
| `t` | int32 | End of the contact interval, in seconds |
| `i` | int32 | Anonymous ID of the first person |
| `j` | int32 | Anonymous ID of the second person |

The interval represented by a row is `(t - 20, t]`. Pair orientation is
preserved from the source; `i` and `j` are not reordered.

The file contains 45,776 rows involving 402 people from the first observation
day. Timestamps range from 32,520 through 77,580 seconds and are aligned to
20-second boundaries. The overnight zero-contact period and following day are
excluded so simulations and inference only cover the active first-day window.

Load it as a pandas DataFrame with:

```python
import pandas as pd

contacts = pd.read_parquet("data/contacts/contacts.parquet")
```

## Rebuild

Requires Python and PyArrow:

```sh
python3 scripts/convert_contacts.py \
  data/contacts/original.dat \
  data/contacts/contacts.parquet \
  --first-day-only
```

The converter validates the source records and writes Zstandard-compressed
Parquet. `--first-day-only` stops before the first contact whose timestamp falls
on the next calendar day; because zero-contact intervals have no rows, the
result ends at the first day's last observed contact.
