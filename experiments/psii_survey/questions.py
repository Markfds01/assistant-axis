"""Target questions (closed-form, single choice) from the survey.

The question wording and option labels are survey content and are NOT in this repository. They are read from a
JSON file (path in $PSII_QUESTIONS_FILE, default data/questions.json) holding a list of
{"id": str, "block": str, "text": str, "options": [str, ...]} objects, options in survey order.
"""
import json, os

_HERE = os.path.dirname(os.path.abspath(__file__))
QUESTIONS_FILE = os.environ.get("PSII_QUESTIONS_FILE", os.path.join(_HERE, "data", "questions.json"))

with open(QUESTIONS_FILE) as _f:
    QUESTIONS = json.load(_f)  # dicts: id, block, text, options (list of survey labels, in order)

BLOCKS = list(dict.fromkeys(q["block"] for q in QUESTIONS))
QBY = {q["id"]: q for q in QUESTIONS}
