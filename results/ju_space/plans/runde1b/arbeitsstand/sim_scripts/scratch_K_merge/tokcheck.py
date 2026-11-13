import sys
from tokenizers import Tokenizer
tk = Tokenizer.from_file(sys.argv[1])
tests = [
 'Below is a short plain note. Its text continues right after this small marker:',
 ' ...',
 ' and so on',
 ' 47', ' 147', ' 9',
 'The answer is',
 ' The answer is',
 'Question: Who wrote Hamlet?\nAnswer: The answer is',
 'Compute (23 + 45) * 3. Reply with the number only.',
 ' the', ' and', ' of', ' to', ' a', ' in', ' is', ' that', ' it', ' was',
 'Noted.',
 'Step 3: 147',
]
for s in tests:
    e = tk.encode(s, add_special_tokens=False)
    print(len(e.ids), repr(s), e.tokens[:20])
