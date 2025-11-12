# S1clean_plus

A **convenient** data-cleaning helper for coursework and real projects:

- Robust numeric coercion (fixes **"age _17" → 17**, "USD 1,234.50" → 1234.50)
- Missing-value imputers: **median / mean / mode / constant / KNN**
- Optional **range validation & clipping** per column (e.g., 0 ≤ age ≤ 120)
- Simple **Kaggle loader helper** (optional extra: `pip install .[kaggle]`)
- **Auto visualizations** of target relationships (matplotlib-only), saved as PNGs
- Clean API: `fit`, `transform`, `fit_transform`, `save_report`
- Only `pandas`, `numpy`, `scikit-learn`, `matplotlib`

## Install

```bash
pip install -e .
# if you want Kaggle loader:
pip install -e .[kaggle]
```
*You need to have `pandas`, `numpy`, `scikit-learn`, `matplotlib` installed in your pc

## Quickstart

```python
import pandas as pd
from s1clean import Cleaner, auto_viz

df = pd.DataFrame({
    "age": [" 17", "age _18", None, "100 years"],
    "city": ["  Yerevan", "YEREVAN", None, "Paris  "],
    "income": ["USD 1,200", "€900.50", "1000", None],
    "joined": ["2024-01-01", "01/02/2024", "not a date", None],
    "target": [1, 0, 1, 0],
})

cl = Cleaner(
    numeric_imputer="knn",            # mean|median|mode|constant|knn
    categorical_imputer="most_frequent",
    outlier_strategy="iqr_cap",
    iqr_k=1.5,
    coerce_mode="strict",             # "strict" extracts numbers from messy text
    validators={
        "age": {"min": 0, "max": 120, "clip": True},
        "income": {"min": 0, "clip": True},
    }
)

clean = cl.fit_transform(df)
cl.save_report("report.html")

# Auto visualizations for the target column:
auto_viz(clean, target="target", out_dir="viz")
```

## Kaggle loader (optional)
```python
from s1clean import load_kaggle_csv

# You must have kaggle API credentials set up: ~/.kaggle/kaggle.json
df = load_kaggle_csv(dataset="zynicide/wine-reviews", file="winemag-data-130k-v2.csv")
```

## Why this is convenient
- Fixes messy numerics without custom regex every time
- Multiple imputation options (including **KNN**)
- Quick sanity checks via validators (clip or flag)
- One-call **EDA visuals** saved to files for reports

