"""
Given a predicted disease name, look up its description, precautions,
medications, and diet recommendations from the supplementary CSVs.
"""

import ast
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _parse_list_string(value):
    """The medications/diets CSVs store values as string-encoded Python lists."""
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, list) else [str(parsed)]
    except (ValueError, SyntaxError):
        return [value]


class DiseaseInfo:
    def __init__(self):
        self.description = pd.read_csv(os.path.join(DATA_DIR, "description.csv"))
        self.precautions = pd.read_csv(os.path.join(DATA_DIR, "precautions_df.csv"))
        self.medications = pd.read_csv(os.path.join(DATA_DIR, "medications.csv"))
        self.diets = pd.read_csv(os.path.join(DATA_DIR, "diets.csv"))
        self.severity = pd.read_csv(os.path.join(DATA_DIR, "Symptom-severity.csv"))

        for df in (self.description, self.precautions, self.medications, self.diets):
            df["Disease"] = df["Disease"].str.strip()

    def get_description(self, disease):
        row = self.description[self.description["Disease"] == disease]
        return row["Description"].values[0] if len(row) else "No description available."

    def get_precautions(self, disease):
        row = self.precautions[self.precautions["Disease"] == disease]
        if not len(row):
            return []
        cols = [c for c in row.columns if c.startswith("Precaution")]
        return [v for v in row.iloc[0][cols].tolist() if pd.notna(v)]

    def get_medications(self, disease):
        row = self.medications[self.medications["Disease"] == disease]
        if not len(row):
            return []
        return _parse_list_string(row["Medication"].values[0])

    def get_diet(self, disease):
        row = self.diets[self.diets["Disease"] == disease]
        if not len(row):
            return []
        return _parse_list_string(row["Diet"].values[0])

    def symptom_severity(self, symptom):
        row = self.severity[self.severity["Symptom"] == symptom]
        return int(row["weight"].values[0]) if len(row) else None
