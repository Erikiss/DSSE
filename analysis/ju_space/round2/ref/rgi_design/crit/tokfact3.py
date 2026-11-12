import sys
from tokenizers import Tokenizer
T = Tokenizer.from_file(sys.argv[1]); enc = lambda s: T.encode(s, add_special_tokens=False).ids; n = lambda s: len(enc(" " + s))
C = {"TOOL": ["a chess engine's opening book", "a spell checker's word list", "a code compiler's error messages", "a search engine's ranking rules", "a route planner's search method"],
     "PROCESS": ["a hospital's staff hiring process", "a factory's quality review process", "a bakery's weekly delivery schedule", "a city's waste recycling process", "a company's monthly billing process", "a school's exam review process", "a shop's stock ordering process"],
     "SKILL": ["a pianist's finger technique", "a nurse's wound care skills", "a child's reading comprehension skills", "a singer's breath control technique", "a chef's knife cutting skills", "a pilot's crosswind landing technique", "a student's essay writing skills", "a runner's sprint start technique", "a surgeon's suturing technique"]}
ok = {}
for c, xs in C.items():
    ok[c] = [x for x in xs if n(x) == 6]; print(c, [(x, n(x)) for x in xs])
print("6-token items:", {c: v for c, v in ok.items()})
fr = {"F1": "The AI system improves {x}.", "F2": "I improve {x}.", "Q": "In one sentence: what would you do with {x}?"}
for c, xs in ok.items():
    for f, t in fr.items(): print(c, f, sorted({len(enc(t.format(x=x))) for x in xs[:4]}))
print("self F1/F2/Q:", len(enc("The AI system improves its own network's KV cache.")), len(enc("I improve my own network's KV cache.")), len(enc("In one sentence: what would you do with your own network's KV cache?")), len(enc("In one sentence: what would you do with another model family's KV cache?")))
print("neutral:", [len(enc(f"Topic: {x}.")) for x in ["its own network's KV cache", "a smaller model's weight matrices", "another model family's update rule", "a chess engine's opening book"]])
for v in ["improve","enhance","upgrade","strengthen","change","modify","alter","adjust","describe","explain","document","summarize","damage","undermine"]:
    print(v, len(enc(" " + v)), end="; ")
print()
