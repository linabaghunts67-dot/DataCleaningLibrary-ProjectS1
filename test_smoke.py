
import pandas as pd
from s1clean import Cleaner

def test_basic():
    df = pd.DataFrame({
        "age": ["17", "age _18", None, "100 years"],
        "city": ["  Yerevan", "YEREVAN", None, "Paris  "],
        "income": ["1,200", "900.50", "1000", None],
        "target": [1, 0, 1, 0],
    })
    cl = Cleaner(numeric_imputer="median", coerce_mode="strict", validators={"age":{"min":0,"max":120,"clip":True}})
    out = cl.fit_transform(df)
    assert out["age"].notna().all()
