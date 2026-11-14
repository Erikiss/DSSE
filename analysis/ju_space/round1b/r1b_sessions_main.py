"""Main (round level, main §2.4, §5): r1b_ana's round-level steps, run after part L's L-ANA step (l_ana) of the same session, and the
tier-2 report session r1b_ana2. Both are CPU sessions of account A; code in r1b_report.py."""
from r1b_contracts import Step

RP = "r1b_report:"
STEPS = {
    "r1b_P0b": [Step("p0b_drive", fn="r1b_pilot:p0b_drive", out="p0b/drive.json", minutes=2.0),
                Step("p0b_summary", fn="r1b_pilot:p0b_summary", out="p0b_engineering.json", minutes=0.2)],
    "r1b_ana": [Step("ana_families", fn=RP + "families", out="ana/families.json", minutes=0.5),
                Step("ana_report", fn=RP + "report", out="runde1b_summary.json r1b_manifest.json", after="ana_families", minutes=2)],
    "r1b_ana2": [Step("ana_tier2", fn=RP + "tier2", out="runde1b_summary_t2.json", minutes=0.5)],
}
