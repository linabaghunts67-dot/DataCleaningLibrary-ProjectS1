
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import re

try:
    from sklearn.impute import KNNImputer
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


_NUM_RE = re.compile(r"[-+]?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?")

def extract_number(x: Any) -> Optional[float]:
    """
    Extract first numeric token from messy text:
    "age _17" -> 17; "USD 1,234.50" -> 1234.50; returns None if none found.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x)
    m = _NUM_RE.search(s.replace("\xa0", " "))
    if not m:
        return np.nan
    token = m.group(0).replace(",", "").replace(" ", "")
    try:
        return float(token)
    except Exception:
        return np.nan

def normalize_cats(s: pd.Series, case: Optional[str] = "lower", strip: bool = True) -> pd.Series:
    s = s.astype("string")
    if strip:
        s = s.str.strip().str.replace(r"\s+", " ", regex=True)
    if case == "lower":
        s = s.str.lower()
    elif case == "upper":
        s = s.str.upper()
    return s

def detect_roles(df: pd.DataFrame, coerce_datetime: bool = True) -> Dict[str, str]:
    roles: Dict[str, str] = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            roles[c] = "numeric"
        else:
            parsed = None
            if coerce_datetime:
                parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
            if coerce_datetime and parsed is not None and parsed.notna().mean() > 0.6:
                roles[c] = "datetime"
            else:
                roles[c] = "categorical"
    return roles

def iqr_caps(x: pd.Series, k: float = 1.5):
    q1 = x.quantile(0.25); q3 = x.quantile(0.75); iqr = q3 - q1
    return float(q1 - k*iqr), float(q3 + k*iqr), float(q1), float(q3), float(iqr)


@dataclass
class Cleaner:
    # imputers
    numeric_imputer: str = "median"         # median|mean|mode|constant|knn
    categorical_imputer: str = "most_frequent"  # most_frequent|constant|mode
    constant_numeric: float = 0.0
    constant_categorical: str = "missing"

    # normalization
    categorical_case: Optional[str] = "lower"
    categorical_strip: bool = True

    # outliers
    outlier_strategy: Optional[str] = "iqr_cap"
    iqr_k: float = 1.5

    # coercion
    coerce_datetime: bool = True
    coerce_mode: str = "loose"   # "loose": to_numeric; "strict": extract_number()

    # validators: per-column rules, e.g. {"age": {"min":0,"max":120,"clip":True}}
    validators: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # learned
    roles_: Dict[str, str] = field(default_factory=dict, init=False)
    num_fill_: Dict[str, Any] = field(default_factory=dict, init=False)
    cat_fill_: Dict[str, Any] = field(default_factory=dict, init=False)
    caps_: Dict[str, Dict[str, float]] = field(default_factory=dict, init=False)
    missing_: Dict[str, float] = field(default_factory=dict, init=False)

    def fit(self, df: pd.DataFrame, y=None):
        df = df.copy()
        self.roles_ = detect_roles(df, self.coerce_datetime)

        # coercion for learning
        for c, r in self.roles_.items():
            if r == "numeric":
                if self.coerce_mode == "strict":
                    df[c] = df[c].map(extract_number).astype(float)
                else:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            elif r == "categorical":
                df[c] = normalize_cats(df[c], self.categorical_case, self.categorical_strip)
            elif r == "datetime" and self.coerce_datetime:
                df[c] = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)

        # learn fill values
        for c, r in self.roles_.items():
            s = df[c]
            if r == "numeric":
                if self.numeric_imputer == "median":
                    self.num_fill_[c] = float(s.median())
                elif self.numeric_imputer == "mean":
                    self.num_fill_[c] = float(s.mean())
                elif self.numeric_imputer in ("mode", "most_frequent"):
                    m = s.mode(dropna=True)
                    self.num_fill_[c] = float(m.iloc[0]) if not m.empty else self.constant_numeric
                elif self.numeric_imputer == "constant":
                    self.num_fill_[c] = self.constant_numeric
                elif self.numeric_imputer == "knn":
                    self.num_fill_[c] = np.nan  # placeholder
                else:
                    self.num_fill_[c] = float(s.median())
            elif r == "categorical":
                if self.categorical_imputer in ("most_frequent","mode"):
                    m = s.mode(dropna=True)
                    self.cat_fill_[c] = str(m.iloc[0]) if not m.empty else self.constant_categorical
                elif self.categorical_imputer == "constant":
                    self.cat_fill_[c] = self.constant_categorical
                else:
                    m = s.mode(dropna=True)
                    self.cat_fill_[c] = str(m.iloc[0]) if not m.empty else self.constant_categorical

        # outlier caps
        self.caps_.clear()
        if self.outlier_strategy == "iqr_cap":
            for c, r in self.roles_.items():
                if r == "numeric":
                    s = df[c].dropna()
                    if not s.empty:
                        lo, hi, q1, q3, iqr = iqr_caps(s, self.iqr_k)
                        self.caps_[c] = {"lower": lo, "upper": hi, "q1": q1, "q3": q3, "iqr": iqr}

        # missingness summary
        self.missing_ = {c: float(df[c].isna().mean()) for c in df.columns}
        return self

    def _apply_validators(self, df: pd.DataFrame):
        for c, rules in self.validators.items():
            if c not in df: continue
            s = df[c]
            lo = rules.get("min", None); hi = rules.get("max", None)
            clip = bool(rules.get("clip", False))
            if lo is not None and hi is not None and clip:
                df[c] = s.clip(lo, hi)
            elif lo is not None and clip:
                df[c] = s.clip(lower=lo)
            elif hi is not None and clip:
                df[c] = s.clip(upper=hi)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.roles_, "Call fit() first."
        df = df.copy()

        # normalize & coerce
        for c, r in self.roles_.items():
            if r == "categorical":
                df[c] = normalize_cats(df[c], self.categorical_case, self.categorical_strip)
            elif r == "numeric":
                if self.coerce_mode == "strict":
                    df[c] = df[c].map(extract_number).astype(float)
                else:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            elif r == "datetime" and self.coerce_datetime:
                df[c] = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)

        # impute categoricals
        for c, r in self.roles_.items():
            if r == "categorical" and c in self.cat_fill_:
                df[c] = df[c].fillna(self.cat_fill_[c])

        # impute numerics
        num_cols = [c for c, r in self.roles_.items() if r == "numeric"]
        if num_cols:
            if self.numeric_imputer == "knn":
                if not SKLEARN_OK:
                    raise RuntimeError("KNN imputer requires scikit-learn. Install the package first.")
                imputer = KNNImputer(n_neighbors=5)
                df[num_cols] = imputer.fit_transform(df[num_cols])
            else:
                for c in num_cols:
                    if c in self.num_fill_:
                        df[c] = df[c].fillna(self.num_fill_[c])

        # outliers (iqr caps)
        if self.outlier_strategy == "iqr_cap":
            for c, caps in self.caps_.items():
                if c in df:
                    df[c] = pd.to_numeric(df[c], errors="coerce").clip(caps["lower"], caps["upper"])

        # validators (clip)
        self._apply_validators(df)

        return df

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(df, y).transform(df)

    def save_report(self, path: str) -> str:
        rows = []
        for c in self.roles_:
            rows.append(
                f"<tr><td><b>{c}</b></td><td>{self.roles_[c]}</td>"
                f"<td>{100*self.missing_.get(c,0):.2f}%</td></tr>"
            )
        cap_rows = []
        for c, caps in self.caps_.items():
            cap_rows.append(
                f"<tr><td>{c}</td><td>{caps['q1']:.4g}</td><td>{caps['q3']:.4g}</td>"
                f"<td>{caps['iqr']:.4g}</td><td>{caps['lower']:.4g}</td><td>{caps['upper']:.4g}</td></tr>"
            )
        html = f"""<!doctype html>
<html lang="en"><meta charset="utf-8">
<title>s1clean_plus report</title>
<style>
  body {{ font-family: system-ui, Segoe UI, Roboto, Arial; margin: 24px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
  th {{ background: #fafafa; }}
  .small {{ color: #666; font-size: 0.9rem; }}
</style>
<h1>Data Cleaning Report</h1>
<div class="small">Generated by <code>s1clean_plus</code></div>

<h2>Columns</h2>
<table>
  <thead><tr><th>Column</th><th>Role</th><th>Missing %</th></tr></thead>
  <tbody>
    {''.join(rows)}
  </tbody>
</table>

{"<h2>Outlier Caps (IQR)</h2><table><thead><tr><th>Col</th><th>Q1</th><th>Q3</th><th>IQR</th><th>Lower</th><th>Upper</th></tr></thead><tbody>"+''.join(cap_rows)+"</tbody></table>" if cap_rows else ""}

</html>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return path


def quick_clean(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return Cleaner(**kwargs).fit_transform(df)
