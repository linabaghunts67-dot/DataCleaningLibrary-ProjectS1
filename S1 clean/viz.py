
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _safe_mkdir(p): os.makedirs(p, exist_ok=True)

def auto_viz(df: pd.DataFrame, target: str, out_dir: str = "viz") -> dict:
    _safe_mkdir(out_dir)
    paths = {"numeric": [], "categorical": [], "target": None}

    # target distribution
    fig = plt.figure()
    if pd.api.types.is_numeric_dtype(df[target]):
        df[target].plot(kind="hist", bins=30)
        plt.title(f"Target histogram: {target}")
        plt.xlabel(target); plt.ylabel("count")
    else:
        df[target].value_counts().plot(kind="bar")
        plt.title(f"Target counts: {target}")
        plt.xlabel(target); plt.ylabel("count")
    p = os.path.join(out_dir, f"target_{target}.png")
    plt.tight_layout(); plt.savefig(p); plt.close(fig)
    paths["target"] = p

    # relationships
    for col in df.columns:
        if col == target: continue
        try:
            if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[target]):
                fig = plt.figure()
                plt.scatter(df[col], df[target], s=10)
                plt.title(f"{col} vs {target}")
                plt.xlabel(col); plt.ylabel(target)
                p = os.path.join(out_dir, f"num_vs_target_{col}.png")
                plt.tight_layout(); plt.savefig(p); plt.close(fig)
                paths["numeric"].append(p)
            elif pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_numeric_dtype(df[target]):
                fig = plt.figure()
                grp = df.groupby(target)[col].mean()
                grp.plot(kind="bar")
                plt.title(f"Mean {col} by {target}")
                plt.xlabel(target); plt.ylabel(f"mean({col})")
                p = os.path.join(out_dir, f"mean_{col}_by_{target}.png")
                plt.tight_layout(); plt.savefig(p); plt.close(fig)
                paths["numeric"].append(p)
            else:
                if pd.api.types.is_numeric_dtype(df[target]):
                    fig = plt.figure()
                    grp = df.groupby(col)[target].mean().sort_values(ascending=False).head(20)
                    grp.plot(kind="bar")
                    plt.title(f"Mean {target} by {col}")
                    plt.xlabel(col); plt.ylabel(f"mean({target})")
                    p = os.path.join(out_dir, f"mean_{target}_by_{col}.png")
                    plt.tight_layout(); plt.savefig(p); plt.close(fig)
                    paths["categorical"].append(p)
                else:
                    fig = plt.figure()
                    ctab = pd.crosstab(df[col], df[target]).head(20)
                    ctab.plot(kind="bar", stacked=True)
                    plt.title(f"{col} vs {target}")
                    plt.xlabel(col); plt.ylabel("count")
                    p = os.path.join(out_dir, f"stack_{col}_vs_{target}.png")
                    plt.tight_layout(); plt.savefig(p); plt.close(fig)
                    paths["categorical"].append(p)
        except Exception:
            plt.close("all")
            continue
    return paths
