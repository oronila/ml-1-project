# Report Model Code

This folder contains only report-friendly code that follows the cleaning pattern
from `sam.ipynb`.

## Main File To Show

- `sam_style_ensemble.py`  
  Self-contained ensemble script using Sam-style preprocessing:
  - drops text/id columns such as `REMARKS`, `COMMENTS`, `LOCATION`, `INDEX_NR`
  - drops `NUM_STRUCK`
  - converts `INCIDENT_DATE` into numeric date features
  - converts `TIME` into minutes plus sine/cosine time features
  - fills categorical missing values with `"Unknown"`
  - frequency-encodes high-cardinality categoricals
  - one-hot encodes low-cardinality categoricals
  - adds missing-value flags
  - median-imputes numeric columns
  - averages simple tree models: HGB, AdaBoost, Random Forest, ExtraTrees

Run:

```bash
python sam_style_ensemble.py
```

Default output:

```text
sam_style_ensemble_submission.csv
```

## Baseline References

- `sam_model.py`  
  A single-model version of the Sam notebook pipeline.

- `original_notebook_model.py`  
  The earlier notebook-style AdaBoost baseline.

The old `feat_pipeline.py`-based files were intentionally removed from this
folder because they do not match the data-cleaning pattern in `sam.ipynb`.
