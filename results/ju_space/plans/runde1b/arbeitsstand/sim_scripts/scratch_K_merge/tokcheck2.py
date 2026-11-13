import sys
from tokenizers import Tokenizer
tk = Tokenizer.from_file(sys.argv[1])
for s in [' and so on.', ' and so on,', ' and so forth', ' and the like', ' as noted', ' as noted here', '\nThe answer is', ' x', ' The', 'Question:', '\nAnswer: The answer is']:
    e = tk.encode(s, add_special_tokens=False)
    print(len(e.ids), repr(s), e.tokens)
