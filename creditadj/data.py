"""Download, verify and cache the UCI "Default of Credit Card Clients" dataset.

Source: Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine
Learning Repository. https://doi.org/10.24432/C55S3H (CC BY 4.0).

The file is never committed. It is downloaded once into ``data/`` and checked
against a SHA-256 digest and the expected shape before anything uses it.
"""

from __future__ import annotations

import hashlib
import io
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
DATASET_NAME = "UCI Default of Credit Card Clients (Yeh & Lien 2009), Taiwan, 2005"

# SHA-256 of the .xls inside the zip, as downloaded in November 2026.
XLS_SHA256 = "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"
N_ROWS = 30_000

XLS_NAME = "default_of_credit_card_clients.xls"
CSV_NAME = "default_of_credit_card_clients.csv"

# Index 1 is September 2005 (most recent), index 6 is April 2005. The source
# file calls the first status column PAY_0 and then jumps to PAY_2. I rename
# it to PAY_1 so the three monthly blocks line up.
PAY_COLS = [f"PAY_{k}" for k in range(1, 7)]
BILL_COLS = [f"BILL_AMT{k}" for k in range(1, 7)]
PAID_COLS = [f"PAY_AMT{k}" for k in range(1, 7)]
PROTECTED_COLS = ["SEX", "AGE", "MARRIAGE", "EDUCATION"]
TARGET = "DEFAULT"

EXPECTED_COLUMNS = (
    ["ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + PAY_COLS
    + BILL_COLS
    + PAID_COLS
    + [TARGET]
)


def default_data_dir() -> Path:
    """``data/`` at the repo root, unless CREDITADJ_DATA_DIR points elsewhere."""
    env = os.environ.get("CREDITADJ_DATA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "data"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(data_dir: Path) -> Path:
    """Fetch the zip from UCI and write the .xls it contains into ``data_dir``."""
    data_dir.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(UCI_URL, headers={"User-Agent": "creditadj"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [m for m in archive.namelist() if m.lower().endswith(".xls")]
        if len(members) != 1:
            raise RuntimeError(f"Expected one .xls in the UCI zip, found {members}")
        xls_bytes = archive.read(members[0])
    target = data_dir / XLS_NAME
    target.write_bytes(xls_bytes)
    return target


def verify_checksum(xls_path: Path) -> None:
    """Fail loudly if the downloaded file is not the one the results were built on."""
    if os.environ.get("CREDITADJ_SKIP_CHECKSUM") == "1":
        return
    actual = _sha256(xls_path)
    if actual != XLS_SHA256:
        raise RuntimeError(
            f"Checksum mismatch for {xls_path}.\n"
            f"  expected {XLS_SHA256}\n  got      {actual}\n"
            "UCI may have republished the file. Set CREDITADJ_SKIP_CHECKSUM=1 to "
            "continue anyway; the row-count and column checks still apply."
        )


def _read_xls(xls_path: Path) -> pd.DataFrame:
    # Row 0 of the sheet holds X1..X23 labels; the real header is row 1.
    frame = pd.read_excel(xls_path, header=1)
    frame = frame.rename(columns={"PAY_0": "PAY_1", "default payment next month": TARGET})
    return frame


def validate(frame: pd.DataFrame) -> None:
    """Shape and content checks that hold for the UCI original."""
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns: {list(frame.columns)}")
    if len(frame) != N_ROWS:
        raise ValueError(f"Expected {N_ROWS} rows, found {len(frame)}")
    if frame.isna().any().any():
        raise ValueError("The UCI file has no missing values; this one does.")
    if not set(frame[TARGET].unique()) <= {0, 1}:
        raise ValueError("Target must be 0/1.")
    if (frame["LIMIT_BAL"] <= 0).any():
        raise ValueError("Credit limits must be positive.")


def load_raw(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Return the 30,000-row dataset, downloading and caching it if needed."""
    data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
    csv_path = data_dir / CSV_NAME
    if csv_path.exists():
        frame = pd.read_csv(csv_path)
        validate(frame)
        return frame

    xls_path = data_dir / XLS_NAME
    if not xls_path.exists():
        try:
            download(data_dir)
        except Exception as exc:  # network errors, proxy errors, bad zip
            raise RuntimeError(
                f"Could not download the dataset from {UCI_URL} ({exc}). "
                f"Download the zip by hand and put the .xls at {xls_path}."
            ) from exc
    verify_checksum(xls_path)
    frame = _read_xls(xls_path)
    validate(frame)
    frame.to_csv(csv_path, index=False)
    return frame


def data_summary(frame: pd.DataFrame) -> dict:
    """Counts of the undocumented codes, so the README can quote real numbers."""
    pay = frame[PAY_COLS].to_numpy()
    bill = frame[BILL_COLS].to_numpy()
    dormant = (bill <= 0).all(axis=1)
    return {
        "rows": int(len(frame)),
        "default_rate": float(frame[TARGET].mean()),
        "education_undocumented_codes": {
            str(code): int((frame["EDUCATION"] == code).sum()) for code in (0, 5, 6)
        },
        "marriage_undocumented_code_0": int((frame["MARRIAGE"] == 0).sum()),
        "pay_status_share_minus2": float((pay == -2).mean()),
        "pay_status_share_minus1": float((pay == -1).mean()),
        "pay_status_share_0": float((pay == 0).mean()),
        "pay_status_share_late": float((pay >= 1).mean()),
        "share_negative_bills": float((bill < 0).mean()),
        "share_bill_over_limit_recent": float((frame["BILL_AMT1"] > frame["LIMIT_BAL"]).mean()),
        # Accounts with no balance on any of the six statements.
        "dormant_accounts": int(dormant.sum()),
        "dormant_default_rate": float(frame.loc[dormant, TARGET].mean()),
    }


def pay_code_evidence(frame: pd.DataFrame) -> list[dict]:
    """What the September status codes look like in the amounts.

    UCI documents -1 and 1 to 9 only. This table is the evidence I used to
    decide what -2 and 0 mean: for each code, how often the September bill is
    zero or negative, the median utilisation, and how often the September
    payment covered the August bill in full.
    """
    limit = frame["LIMIT_BAL"].to_numpy(dtype=float)
    bill_sep = frame["BILL_AMT1"].to_numpy(dtype=float)
    bill_aug = frame["BILL_AMT2"].to_numpy(dtype=float)
    paid_sep = frame["PAY_AMT1"].to_numpy(dtype=float)
    status = frame["PAY_1"].to_numpy()
    target = frame[TARGET].to_numpy()

    rows = []
    for label, mask in [
        ("-2", status == -2),
        ("-1", status == -1),
        ("0", status == 0),
        ("1", status == 1),
        ("2 or more", status >= 2),
    ]:
        rows.append(
            {
                "code": label,
                "n": int(mask.sum()),
                "default_rate": float(target[mask].mean()),
                "share_bill_zero_or_negative": float((bill_sep[mask] <= 0).mean()),
                "median_utilisation": float(np.median(bill_sep[mask] / limit[mask])),
                "share_paid_august_bill_in_full": float((paid_sep[mask] >= bill_aug[mask]).mean()),
            }
        )
    return rows
