# Contact data

`contacts.parquet` is an analysis-ready conversion of `original.dat`. Each row
represents one pair observed during a one-minute interval.

## Schema

| Column | Type | Meaning |
| --- | --- | --- |
| `t` | int32 | End of the contact interval, in seconds |
| `i` | int32 | Anonymous ID of the first person |
| `j` | int32 | Anonymous ID of the second person |

The interval represented by a row is `(t - 60, t]`. A pair is included when it
appears in at least one of the source's three 20-second intervals in that
minute. Repeated and reverse-oriented observations of the same pair in a
minute are collapsed to one row, while the first observed orientation is
preserved.

The file contains 25,057 rows involving 402 people from the first observation
day. Timestamps range from 32,520 through 77,580 seconds and are aligned to
one-minute boundaries. The overnight zero-contact period and following day are
excluded so simulations and inference only cover the active first-day window.

Load it as a pandas DataFrame with:

```python
import pandas as pd

contacts = pd.read_parquet("data/contacts/contacts.parquet")
```

## Rebuild

Requires Python and PyArrow:

```sh
python3 data/convert_contacts.py \
  data/contacts/original.dat \
  data/contacts/contacts.parquet \
  --first-day-only
```

The converter validates the source records and writes Zstandard-compressed
Parquet. `--first-day-only` stops before the first contact whose timestamp falls
on the next calendar day; because zero-contact intervals have no rows, the
result ends at the first day's last observed contact.
