"""SO-GVRA (main §2.3, §2.5; owner K): the inverse-free generate-verify-revise-abstain loop. A generator proposes a candidate; a
verifier (a forward round trip through the unmodified model, decoupled from the generator) returns 'correct', 'minor fix needed' or
'critically flawed'; a reviser fixes minor flaws; at most `limit` attempts per unit, then the unit abstains. The verifier's error rate
is measured on known cases before its outputs are read (error_rates)."""
OUTCOMES = ("correct", "minor fix needed", "critically flawed")


def gvra(units, generate, verify, revise=None, limit=3):
    """units: iterable of unit ids; generate(unit, attempt) -> candidate or None (exhausted); verify(unit, cand) -> (outcome, score);
    revise(unit, cand, attempt) -> candidate. Returns per unit dict(status 'accepted' | 'abstained', candidate, score, attempts, trace)."""
    out = {}
    for u in units:
        trace, acc, a, cand = [], None, 0, None
        while a < limit:
            cand = generate(u, a) if (cand is None or not trace or trace[-1][0] == "critically flawed") else cand
            if cand is None:
                break
            o, s = verify(u, cand); trace.append((o, s)); a += 1
            if o == "correct":
                acc = (cand, s); break
            if o == "minor fix needed" and revise is not None and a < limit:
                cand = revise(u, cand, a)
                o2, s2 = verify(u, cand); trace.append((o2, s2)); a += 1
                if o2 == "correct":
                    acc = (cand, s2); break
                cand = None if o2 == "critically flawed" else cand
            else:
                cand = None
            if cand is None:
                trace.append(("critically flawed", None)) if trace[-1][0] != "critically flawed" else None
        out[u] = dict(status="accepted" if acc else "abstained", candidate=acc[0] if acc else None, score=acc[1] if acc else None,
                      attempts=a, trace=trace)
    return out


def error_rates(pass_true, pass_false):
    """Verifier error on known cases: sensitivity = share of true candidates accepted; false-accept = share of foreign ones accepted."""
    n1, n0 = max(len(pass_true), 1), max(len(pass_false), 1)
    return dict(sensitivity=sum(map(bool, pass_true)) / n1, false_accept=sum(map(bool, pass_false)) / n0, n_true=len(pass_true),
                n_false=len(pass_false))


def summary(res):
    acc = sum(r["status"] == "accepted" for r in res.values())
    return dict(n=len(res), accepted=acc, abstained=len(res) - acc, attempts=sum(r["attempts"] for r in res.values()))
