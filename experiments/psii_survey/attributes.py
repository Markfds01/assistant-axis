"""Identity attributes and their value sets for the survey.

Each attribute maps raw survey labels to a small set of
values. Labels not listed (e.g. "I prefer not to answer", NaN) map to None: the respondent
gets no value (and no sentence/vector) for that attribute.
"""
import pandas as pd

ATTRIBUTES = ["sex", "age", "education", "income", "religion", "marital_status", "politics"]

# Raw survey labels -> value keys are survey content and are not committed. They are read from a JSON file
# ($PSII_LABEL_MAPS, default data/label_maps.json): {"education": {raw label: value key}, "income": ..., "religion": ...,
# "marital_status": ..., "politics": ...}.
import json, os
_HERE = os.path.dirname(os.path.abspath(__file__))
LABEL_MAPS_FILE = os.environ.get("PSII_LABEL_MAPS", os.path.join(_HERE, "data", "label_maps.json"))
with open(LABEL_MAPS_FILE) as _f:
    _MAPS = json.load(_f)

# value -> natural-language description used in the profile prompt AND in the persona-generation prompts
VALUE_DESC = {
    "sex": {"Male": "a man", "Female": "a woman"},
    "age": {"18_29": "aged between 18 and 29 (young adult)", "30_44": "aged between 30 and 44",
            "45_59": "aged between 45 and 59", "60_plus": "aged 60 or over"},
    "education": {"primary_or_less": "primary education or less (or no diploma)",
                  "lower_secondary": "compulsory secondary education (ESO)",
                  "upper_secondary_or_vocational": "Bachillerato or vocational training",
                  "bachelors": "a university bachelor's degree",
                  "postgraduate": "a master's degree or doctorate"},
    "income": {"le_1100": "a monthly household income of 1,100 euros or less",
               "1101_1700": "a monthly household income between 1,101 and 1,700 euros",
               "1701_2400": "a monthly household income between 1,701 and 2,400 euros",
               "2401_3500": "a monthly household income between 2,401 and 3,500 euros",
               "gt_3500": "a monthly household income above 3,500 euros"},
    "religion": {"catholic": "Catholic", "none": "non-religious (atheist or agnostic)",
                 "other_christian": "a Christian of a non-Catholic denomination (Protestant, Orthodox or other)",
                 "muslim": "Muslim", "other_religion": "an adherent of another religion (e.g. Buddhist, Hindu, Jewish)"},
    "marital_status": {"married": "married or in a civil union", "never_married": "never married",
                       "divorced_separated": "divorced or separated", "widowed": "widowed"},
    "politics": {"PSOE": "closest to the Spanish party PSOE (social democratic, centre-left)",
                 "PP": "closest to the Spanish party PP (centre-right, conservative)",
                 "Vox": "closest to the Spanish party Vox (radical right)",
                 "Sumar": "closest to the Spanish party Sumar (left-wing)",
                 "other_or_regional": "closest to a regional or other minor party (e.g. Junts, PNV, EH Bildu, BNG, CC, UPN)"},
}
VALUES = {a: list(v) for a, v in VALUE_DESC.items()}



def age_band(x):
    x = pd.to_numeric(x, errors="coerce")
    if pd.isna(x):
        return None
    return "18_29" if x <= 29 else "30_44" if x <= 44 else "45_59" if x <= 59 else "60_plus"


def map_value(attr, raw):
    if attr == "age":
        return age_band(raw)
    if attr == "sex":
        return raw if raw in ("Male", "Female") else None
    return _MAPS[attr].get(raw)


def respondent_identity(row):
    return {a: map_value(a, row[a]) for a in ATTRIBUTES}
