"""Definitions fixed by the plan review of README section 9 (act_plan_review.json, rule 'Definitions'), applied to log.json of kv_grok.py.
event        one or more consecutive evaluations (50 steps apart) at which the weight norm is at least 0.5 higher than at the
             previous evaluation, or the batch loss is at least 0.1, or the accuracy on the training problems of the run is below 0.99
levels       early = median of the held-out accuracy over the evaluations of steps 1000 to 1500, late = over steps 3500 to 4000,
             end = over steps 7500 to 8000;  t50 = first evaluation with held-out accuracy below 0.50
readable     a drift control is readable if its held-out accuracy is at least 0.99 at 90 % or more of its evaluations and its late level is at least 0.99
trace events for runs evaluated less often than every 50 steps (lr 3e-4: every 150 steps): from the trace of batch loss and weight
             norm that kv_grok.py stores every 50 steps; an event is a trace point with batch loss at least 0.1 or a weight norm at
             least 0.5 above the previous trace point (consecutive points merged)"""
import json, statistics


def load_log(path):
    J = json.load(open(path)); return J, J["log"]


def events(log, upto=None):
    """List of (first step, last step) of the events of a run (evaluations must be 50 steps apart)."""
    ev, cur, prev = [], None, None
    for r in log:
        if upto is not None and r["step"] > upto: break
        hit = (prev is not None and r["wnorm"] - prev["wnorm"] >= 0.5) or r["task"] >= 0.1 or r["train_acc"] < 0.99
        if hit: cur = [r["step"], r["step"]] if cur is None else [cur[0], r["step"]]
        elif cur is not None: ev.append(tuple(cur)); cur = None
        prev = r
    if cur is not None: ev.append(tuple(cur))
    return ev


def trace_events(trace):
    ev, cur = [], None
    for a, t in zip([None] + trace[:-1], trace):
        hit = t[1] >= 0.1 or (a is not None and t[2] - a[2] >= 0.5)
        if hit: cur = [t[0], t[0]] if cur is None else [cur[0], t[0]]
        elif cur is not None: ev.append(tuple(cur)); cur = None
    if cur is not None: ev.append(tuple(cur))
    return ev


def level(log, lo, hi):
    v = [r["val_acc"] for r in log if lo <= r["step"] <= hi]; return statistics.median(v) if v else None


def t50(log): return next((r["step"] for r in log if r["val_acc"] < 0.5), None)


def summary(path):
    J, log = load_log(path); ev = events(log); t = t50(log); n = len(log)
    return dict(steps=log[-1]["step"], eval_every=log[1]["step"] - log[0]["step"] if n > 1 else None, early=level(log, 1000, 1500), late=level(log, 3500, 4000), end=level(log, 7500, 8000), t50=t, events=ev, events_first_1500=len([e for e in ev if e[0] <= 1500]),
                events_before_t50=len([e for e in ev if t is None or e[0] < t]), min_held=min(r["val_acc"] for r in log), share_held_099=sum(r["val_acc"] >= 0.99 for r in log) / n, share_train_099=sum(r["train_acc"] >= 0.99 for r in log) / n, min_train=min(r["train_acc"] for r in log),
                wnorm_first=log[0]["wnorm"], wnorm_last=log[-1]["wnorm"], task_last=log[-1]["task"], task_median=statistics.median(r["task"] for r in log), held_at={str(s): next((r["val_acc"] for r in log if r["step"] == s), None) for s in (250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000)},
                trace_events=trace_events(J["trace"]) if J.get("trace") else None, readable_as_drift=bool(sum(r["val_acc"] >= 0.99 for r in log) / n >= 0.9 and (level(log, 3500, 4000) or 0) >= 0.99), config={k: J.get(k) for k in ("mode", "seed", "sub", "lr", "eps", "amsgrad", "batch", "frozen", "init")})
