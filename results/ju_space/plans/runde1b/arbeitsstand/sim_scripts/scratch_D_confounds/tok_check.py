"""Tokenizer-only checks for the part-D spec (no model is loaded or run).
Measures: canonical chars/token on repo wikitext passages, character-level inflation,
random non-canonical segmentation inflation, single-token status of candidate value words,
token lengths of fixed template pieces."""
import json, lzma, random, sys
from tokenizers import Tokenizer
TOKP = sys.argv[1]; CORP = sys.argv[2]
tk = Tokenizer.from_file(TOKP)
enc = lambda s: tk.encode(s, add_special_tokens=False).ids
data = json.loads(lzma.open(CORP).read().decode())
texts = data if isinstance(data, list) else data.get('texts', data.get('passages', []))
if isinstance(texts, dict): texts = list(texts.values())
texts = [t if isinstance(t, str) else t.get('text', '') for t in texts]
texts = [t for t in texts if len(t) > 200]
print('passages', len(texts))
nch = sum(len(t) for t in texts); ntok = sum(len(enc(t)) for t in texts)
nbytes = sum(len(t.encode('utf-8')) for t in texts)
print(f'canonical: chars {nch} tokens {ntok} chars/token {nch/ntok:.3f} bytes/token {nbytes/ntok:.3f}')
vocab = tk.get_vocab()
# character-level: one token per character, if each char (as byte-level token) exists
def char_ids(t):
    out = []
    for ch in t:
        ids = enc(ch)
        out.extend(ids)
    return out
nchar = sum(len(char_ids(t)) for t in texts[:10]); nc10 = sum(len(enc(t)) for t in texts[:10])
print(f'char-level inflation (10 passages): {nchar/nc10:.3f}x')
# random non-canonical segmentation: split each word at random points into pieces, encode pieces separately
rng = random.Random(0)
def rand_seg_ids(t):
    out = []
    for w in t.split(' '):
        pieces = []
        i = 0
        while i < len(w):
            j = min(len(w), i + rng.randint(1, 4)); pieces.append(w[i:j]); i = j
        first = True
        for k, p in enumerate(pieces):
            s = (' ' if not first or out else '') + p if k == 0 else p
            out.extend(enc(s)); first = False
    return out
nrs = sum(len(rand_seg_ids(t)) for t in texts[:10])
print(f'random-segmentation inflation (10 passages, pieces 1-4 chars): {nrs/nc10:.3f}x')
cands = {'colour': ['red','blue','green','yellow','white','black','grey','brown','orange','purple','pink','gold','silver','violet','crimson','teal'],
         'material': ['oak','pine','stone','brick','iron','copper','glass','clay','bronze','marble','slate','tin','wool','linen','silk','ash'],
         'animal': ['fox','owl','bear','wolf','hawk','deer','horse','swan','crow','hare','lynx','otter','raven','stag','boar','heron'],
         'weekday': ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
         'month': ['January','February','March','April','May','June','July','August','September','October','November','December']}
for k, ws in cands.items():
    single = [w for w in ws if len(enc(' ' + w)) == 1]; multi = [w for w in ws if len(enc(' ' + w)) != 1]
    print(f'{k}: single-token {len(single)}/{len(ws)}; multi: {multi}')
SUF = "<|im_end|>\n<|im_start|>user\nContinue.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
print('TURN2_SUFFIX tokens', len(tk.encode(SUF, add_special_tokens=False).ids))
print('empty think block tokens', len(tk.encode('<think>\n\n</think>\n\n', add_special_tokens=False).ids))
