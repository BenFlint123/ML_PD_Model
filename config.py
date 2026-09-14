from pathlib import Path

import pandas as pd


# Paths
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / ".git").exists())
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

# Pandas
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)

# General params:
CATEGORICAL_THRESHOLD = 25
RANDOM_SEED = 42

# Variable names
TARGET_VARIABLE = "Y_DEF_FLAG_1M"
ID_VARIABLE = "ID"
CAT_VARS = ["X02_SEX", "X03_EDUCATION", "X04_MARRIAGE"]
NUM_VARS = [
    "X01_LIMIT_BAL",
    "X05_AGE",
    "X12_BILL_AMT1",
    "X13_BILL_AMT2",
    "X14_BILL_AMT3",
    "X15_BILL_AMT4",
    "X16_BILL_AMT5",
    "X17_BILL_AMT6",
    "X18_PAY_AMT1",
    "X19_PAY_AMT2",
    "X20_PAY_AMT3",
    "X21_PAY_AMT4",
    "X22_PAY_AMT5",
    "X23_PAY_AMT6",
    "X06_PAY_0",
    "X07_PAY_2",
    "X08_PAY_3",
    "X09_PAY_4",
    "X10_PAY_5",
    "X11_PAY_6",
]

# 00_EDA_Taiwan_dataset.ipynb specific params:
COL_NAME_MAP = {
    "Unnamed: 0_ID": "ID",
    "X1_LIMIT_BAL": "X01_LIMIT_BAL",
    "X2_SEX": "X02_SEX",
    "X3_EDUCATION": "X03_EDUCATION",
    "X4_MARRIAGE": "X04_MARRIAGE",
    "X5_AGE": "X05_AGE",
    "X6_PAY_0": "X06_PAY_0",
    "X7_PAY_2": "X07_PAY_2",
    "X8_PAY_3": "X08_PAY_3",
    "X9_PAY_4": "X09_PAY_4",
    "Y_default payment next month": "Y_DEF_FLAG_1M",
}
EDUCATION_MAP = {0: 4, 5: 4, 6: 4}
MARRIAGE_MAP = {0: 3}
