"""Export Table S1 and return its table and sex-ratio test for notebook display."""

from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def export_table_s1(info_file=None, output_dir=None):
    info_file = Path(info_file) if info_file is not None else ROOT / "00_data/participants.csv"
    output_dir = Path(output_dir) if output_dir is not None else ROOT / "results/SI"
    info = pd.read_csv(info_file, encoding="utf-8-sig", dtype={"ID": str})
    info["group"] = pd.cut(info.Age_group, [4, 6, 8, 10], labels=["5-6", "7-8", "9-10"])
    rows = []
    for group, data in list(info.groupby("group", observed=True)) + [("All", info)]:
        rows.append(
            {
                "Age group": group,
                "n": len(data),
                "Boys/Girls": f"{(data.Gender == 'M').sum()}/{(data.Gender == 'F').sum()}",
                "Age (months), M ± SD": f"{data.Age_month.mean():.1f} ± {data.Age_month.std():.1f}",
                "Age (months), range": f"{data.Age_month.min()}–{data.Age_month.max()}",
                "Age (years), M ± SD": f"{data.Age_month.mean() / 12:.2f} ± {data.Age_month.std() / 12:.2f}",
            }
        )
    sex_chi = stats.chi2_contingency(pd.crosstab(info.group, info.Gender))
    table = pd.DataFrame(rows)
    note = f"Sex ratio by age group: chi2({sex_chi[2]}) = {sex_chi[0]:.2f}, p = {sex_chi[1]:.3f}\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "Table_S1_sample_characteristics.csv", index=False)
    (output_dir / "Table_S1_note.txt").write_text(note, encoding="utf-8")
    return table, note


if __name__ == "__main__":
    table, note = export_table_s1()
    print(table.to_string(index=False))
    print(note)
