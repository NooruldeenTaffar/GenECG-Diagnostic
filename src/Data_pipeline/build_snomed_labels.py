from pathlib import Path
import pandas as pd
import ast
import csv
from collections import Counter

RAW_DIR = Path("data/Raw/PTBXL")
PTBXL_DB = RAW_DIR / "ptbxl_database.csv"
PTBXL_TO_SNOMED = RAW_DIR / "ptbxlToSNOMED.csv"
OUT = RAW_DIR / "ptbxl_with_snomed.csv"

def read_ptbxl_to_snomed(path: Path) -> pd.DataFrame:
    """
    Some releases store each row as a single quoted string like:
    "Acronym,Dx Statement,id1,name1,..."
    In that case pandas sees ONE column. We detect and fix it using csv.reader.
    """
    # First try normal parsing
    try:
        m = pd.read_csv(path, engine="python")
        if len(m.columns) >= 3:
            return m
    except Exception:
        pass

    # If still one column, parse manually
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)  # respects quotes properly
        for row in reader:
            # row is a list already split into columns
            if len(row) > 1:
                rows.append(row)

    if not rows:
        raise RuntimeError("Could not parse ptbxlToSNOMED.csv. File may be corrupted or empty.")

    header = [h.strip() for h in rows[0] if h is not None]
    data = rows[1:]
    m = pd.DataFrame(data, columns=header)

    # Drop empty trailing columns if header ends with ""
    m = m.loc[:, [c for c in m.columns if str(c).strip() != ""]]
    return m

def main():
    # Load PTB-XL database (ecg_id + scp_codes)
    df = pd.read_csv(PTBXL_DB)
    df["scp_codes_dict"] = df["scp_codes"].apply(ast.literal_eval)

    # Load mapping Acronym -> SNOMED (id1)
    m = read_ptbxl_to_snomed(PTBXL_TO_SNOMED)
    m.columns = [str(c).strip() for c in m.columns]

    print("Mapping CSV columns:", list(m.columns))

    if "Acronym" not in m.columns or "id1" not in m.columns:
        raise RuntimeError("ptbxlToSNOMED.csv must contain columns: 'Acronym' and 'id1'")

    # Build mapping
    scp_to_snomed = {}
    for _, r in m.iterrows():
        scp = str(r["Acronym"]).strip()
        snomed = str(r["id1"]).strip()
        if scp and snomed and snomed.lower() != "nan":
            scp_to_snomed[scp] = snomed

    # Build per-ECG SNOMED codes
    snomed_lists = []
    counts = Counter()

    for d in df["scp_codes_dict"]:
        codes = []
        for scp_code in d.keys():  # acronyms like 'NDT', 'LAFB', ...
            if scp_code in scp_to_snomed:
                codes.append(scp_to_snomed[scp_code])
        codes = sorted(set(codes))
        snomed_lists.append(codes)
        counts.update(codes)

    df["snomed_codes"] = snomed_lists
    df.to_csv(OUT, index=False)

    non_empty = sum(1 for x in snomed_lists if len(x) > 0)
    print(f"Saved: {OUT}")
    print("Non-empty SNOMED rows:", non_empty, "/", len(df))
    print("Example snomed_codes (first non-empty):", next((x for x in snomed_lists if x), []))
    print("Top 10 SNOMED codes:", counts.most_common(10))

if __name__ == "__main__":
    main()
