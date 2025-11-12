
from typing import Optional
import pandas as pd

def load_kaggle_csv(dataset: str, file: str, *, version: Optional[int]=None, path: str="kaggle_data") -> pd.DataFrame:
    """
    Load a CSV from Kaggle using the official API.
    Requirements:
      - pip install kaggle
      - have ~/.kaggle/kaggle.json with your API credentials
    Args:
      dataset: e.g., "zynicide/wine-reviews"
      file:    e.g., "winemag-data-130k-v2.csv"
      version: optional version number
      path:    local folder to download into
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        raise RuntimeError("Kaggle API not installed. Install extra: pip install .[kaggle]") from e

    api = KaggleApi(); api.authenticate()
    if version is None:
        api.dataset_download_file(dataset=dataset, file_name=file, path=path, force=False, quiet=True)
    else:
        api.dataset_download_file(dataset=dataset, file_name=file, path=path, force=False, quiet=True, dataset_version_number=version)
    import os, zipfile
    zip_path = f"{path}/{file}.zip"
    csv_path = f"{path}/{file}"
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(path)
        try: os.remove(zip_path)
        except Exception: pass
    return pd.read_csv(csv_path)
