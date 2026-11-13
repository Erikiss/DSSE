import sys, re, textwrap
p = sys.argv[1]; lines = open(p).read().split("\n")
W = 134
out = []; i = 0
bullet = re.compile(r"^(\s*)- ")
def is_special(l):
    s = l.lstrip()
    return l.strip() == "" or s.startswith("|") or s.startswith("#") or (s.startswith("**") and l == s)
while i < len(lines):
    l = lines[i]
    if is_special(l): out.append(l); i += 1; continue
    m = bullet.match(l)
    if m:
        ind = len(m.group(1)); text = l.strip()[2:]; i += 1
        while i < len(lines) and not is_special(lines[i]) and not bullet.match(lines[i]) and (len(lines[i]) - len(lines[i].lstrip())) > ind:
            text += " " + lines[i].strip(); i += 1
        out.extend(textwrap.wrap(text, width=W, initial_indent=" " * ind + "- ", subsequent_indent=" " * (ind + 2),
                                 break_long_words=False, break_on_hyphens=False))
        continue
    # plain paragraph (column 0 or indented non-bullet)
    ind = len(l) - len(l.lstrip()); text = l.strip(); i += 1
    while i < len(lines) and not is_special(lines[i]) and not bullet.match(lines[i]):
        text += " " + lines[i].strip(); i += 1
    out.extend(textwrap.wrap(text, width=W, initial_indent=" " * ind, subsequent_indent=" " * ind, break_long_words=False, break_on_hyphens=False))
s = "\n".join(out)
open(p, "w").write(s)
print("lines", len(out), "max non-table line", max(len(x) for x in out if not x.lstrip().startswith("|")))
