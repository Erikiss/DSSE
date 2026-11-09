"""List every context shot (96 k16 contexts) hit by the broad brevity regex, outside 08152, to judge by eye whether it is a brevity request."""
import sys, re
sys.path.insert(0, "/tmp/claude-0/-home-user-DSSE/85db304a-b0fa-5d03-b685-f9acf8fac842/scratchpad")
from load_runs import shots_for, DATASET_TABLE
BREV_W = re.compile(r"(?i)\bshort\b|\bbrief|\bquick(?:ly)?\b|concise|one[- ](?:word|line|sentence)|in a (?:word|sentence|nutshell)|"
                    r"bottom line|tl;?dr|\bjust (?:tell|give|say|the)|\bsimpl(?:e|y) (?:answer|yes|no)|yes or no|straight answer|"
                    r"don'?t (?:need|want) (?:a )?(?:long|lengthy|essay|lecture|details|explanation)|no (?:long|lengthy|essay|lecture)|"
                    r"keep it|in short|takeaway|summar|cut to|get to the point|\bsuccinct|\bterse|few words|without (?:the )?(?:details|explanation)")
for d in DATASET_TABLE.dataset_id:
    if d == "dsse1-08152":
        continue
    for s in range(3):
        for i, x in enumerate(shots_for(d, s, 16)):
            m = BREV_W.search(x["prompt"])
            if m:
                print(f"{d} s{s} #{i} [{m.group()}] rw={len(x['response'].split())} | {x['prompt'][:150]!r}")
