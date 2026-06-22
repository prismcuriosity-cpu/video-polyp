# Datasets

PolypAI uses a **modular dataset registry** (`polypai.data.registry`) so sources
can be swapped, relocated, or added by editing config — not code. A registered
source is a function `(root, split) -> list[Sample]`; a `Sample` carries an
image path, an optional mask, optional boxes, and a partial label dict
(`paris`/`nice`/`kudo`/`malignancy`). Missing fields are fine — the trainer masks
them per task, so datasets with different annotation levels train together.

## Supported sources

| Name (registry key) | Annotations | Expected layout (under `root`) |
|---|---|---|
| `kvasir_seg` | masks | `images/*.jpg`, `masks/*.jpg` (matched by stem) |
| `piccolo` | masks + Paris/NICE, WL+NBI | `<split>/polyps/*`, `<split>/masks/*` |
| `cvc_clinicdb` | masks | `Original/*`, `Ground Truth/*` |
| `cvc_colondb` | masks | `images/*`, `masks/*` |
| `hyperkvasir_seg` | masks | `images/*`, `masks/*` |
| `gastrovision` | class labels (placeholder) | `<class_name>/*.jpg` |

The loaders are tolerant of common folder-name variants (case-insensitive,
`image`/`images`, `Ground Truth`/`masks`, …).

## Pointing the pipeline at your data

Edit the `data.datasets` map in your config (e.g. `configs/experiment_rtx5090.yaml`):

```yaml
data:
  datasets:
    kvasir_seg: /data/Kvasir-SEG
    piccolo:    /data/PICCOLO
    # cvc_clinicdb: /data/CVC-ClinicDB
```

> **Placeholders are safe.** If a `root` does not exist, `resolve_samples`
> returns 0 samples with a warning instead of crashing — so you can list
> not-yet-downloaded datasets in a config and add the data later. This is how the
> "insert dataset links later" requirement is met.

## Download links (fill in for your environment)

These are public research datasets; insert the URLs/credentials appropriate to
your institution and license terms:

- **Kvasir-SEG** — `<insert link>`
- **PICCOLO** — `<insert link>` (registration required)
- **CVC-ClinicDB / CVC-ColonDB** — `<insert link>`
- **HyperKvasir** — `<insert link>`
- **GastroVision** — `<insert link>`

## Splits

Datasets without an official split use a **deterministic per-filename hash
split** (80/10/10). For publication this must be upgraded to **per-patient**
splits to prevent frame-level leakage (see `ROADMAP.md` §4); wire patient IDs
into `Sample.meta` and group by them in the loader.

## Adding a new dataset

```python
from polypai.data.registry import register_dataset, Sample

@register_dataset("my_set", has_masks=True, label_keys=("nice",))
def load_my_set(root, split):
    samples = []
    for img in (root / "frames").glob("*.png"):
        samples.append(Sample(
            image_path=str(img),
            mask_path=str(root / "masks" / img.name),
            labels={"nice": int(...)},        # optional, per task
        ))
    return samples
```

Import side-effects in `polypai/data/sources.py` register built-ins at package
import; add new ones there (or anywhere imported before training).

## Detection boxes from masks

Datasets that ship only masks still train the detector: `PolypDataset` derives
axis-aligned boxes from mask connected components (`mask_to_boxes`), one per
lesion, so segmentation supervision doubles as detection supervision.
