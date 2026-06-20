# shapecompare

Python library for identifying physical objects by shape. Give it an image, get back a contour descriptor you can compare against a database of known shapes.

Works for any 2D object — hangers, keys, coins, tools, anything with a recognisable silhouette.

## What it does

1. **Extract** — load an image and pull out the dominant contour
2. **Describe** — convert the contour into a compact descriptor vector
3. **Match** — rank a list of template descriptors by similarity to the query

## What it doesn't do

- No database — you store descriptors wherever you want
- No web UI — wire it into your own app
- No domain-specific logic — it's shape math only

## Dependencies

```
numpy
opencv-python
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Building a template (storing a known shape)

```python
import shapecompare as sc

img = sc.load_image("template.jpg")
result = sc.extract_main_contour(img)

sampled = sc.sample_contour_uniform(result.contour)
fwd_desc = sc.build_descriptor(sampled, result.perimeter)
rev_desc = sc.build_descriptor(sc.reverse_sampled(sampled), result.perimeter)

# Store in your DB: (id, fwd_desc.tobytes(), rev_desc.tobytes())
```

### Querying (identifying an unknown shape)

```python
import numpy as np
import shapecompare as sc

img = sc.load_image_bytes(raw_bytes)  # or sc.load_image("query.jpg")
result = sc.extract_main_contour(img)

sampled = sc.sample_contour_uniform(result.contour)
query_desc = sc.build_descriptor(sampled, result.perimeter)

# Load templates from your DB as list of (id, fwd_desc, rev_desc)
templates = [
    (row["id"], np.frombuffer(row["fwd"], dtype=np.float32), np.frombuffer(row["rev"], dtype=np.float32))
    for row in db_rows
]

matches = sc.match(query_desc, templates, top_k=5)
# matches[0].id      → best match id
# matches[0].distance → how close (lower = better)
```

## Invariances

The descriptor is invariant to:

- **Translation** — pairwise distances, not absolute coordinates
- **Scale** — distances normalised by perimeter
- **Rotation** — comparison tries all cyclic shifts
- **Mirror/flip** — both forward and reversed descriptors are compared

## Image input types

`extract_main_contour` auto-detects the background type:

- Dark object on light background (line drawings, white-paper photos)
- Light/silver object on dark background (product photos on black)
- Outline drawings — interior is flood-filled automatically
