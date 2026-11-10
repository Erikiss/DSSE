#!/usr/bin/env python3
"""Baut die Infografik zu results/ju_space (eine HTML-Seite, deutsch). Alle Zahlen sind aus results/ju_space/README.md
uebernommen; die Diagramme werden hier aus diesen Zahlen massstaeblich erzeugt.
Aufruf: python infografik_build.py <ausgabe.html> [--standalone]
  ohne --standalone: Seitenfragment (ohne <html>/<head>/<body>), wie es als Artifact veroeffentlicht wird
  mit  --standalone: vollstaendiges HTML-Dokument zum Oeffnen im Browser"""
import html, math, re, sys


def esc(s):
    return html.escape(s, quote=True)


def num(x):
    return f"{x:.5g}"


# ----------------------------------------------------------------------------------------------------------------------
# Bausteine
# ----------------------------------------------------------------------------------------------------------------------
def bars(rows, base=0.0, top=100.0, vw="4.4rem", ticks=(), axis=(), cls=""):
    """Waagerechte Balken. rows: dicts mit lab, optional sub, tip, bars=[dict(v, disp, cls, rng=(lo, hi), ext=wert)]
    oder dict(grp=Titel). base/top: Wert am linken und rechten Rand der Zeichenflaeche."""
    m = top - base
    out = [f'<div class="bars {cls}" style="--m:{num(m)};--vw:{vw}">']
    for r in rows:
        if "grp" in r:
            out.append(f'<div class="grp">{r["grp"]}</div>')
            continue
        tip = f' data-tip="{esc(r["tip"])}" tabindex="0"' if r.get("tip") else ""
        lab = r["lab"] + (f'<small>{r["sub"]}</small>' if r.get("sub") else "")
        tracks = []
        for b in r["bars"]:
            v = b["v"] - base
            if b.get("rng"):
                lo, hi = b["rng"][0] - base, b["rng"][1] - base
                e = b.get("ext", b["v"]) - base
                inner = f'<span class="bar {b["cls"]}" style="width:{100 * v / hi:.2f}%"></span>'
                if e > v:
                    inner += f'<span class="ext {b["cls"]}" style="left:{100 * v / hi:.2f}%;width:{100 * (e - v) / hi:.2f}%"></span>'
                inner += f'<span class="whisk" style="left:{100 * lo / hi:.2f}%;width:{100 * (hi - lo) / hi:.2f}%"></span>'
                tracks.append(f'<div class="track"><span class="wrap" style="--v:{num(hi)}">{inner}</span><span class="val">{b["disp"]}</span></div>')
            else:
                tracks.append(f'<div class="track"><span class="bar {b["cls"]}" style="--v:{num(v)}"></span><span class="val">{b["disp"]}</span></div>')
        tk = "".join(f'<span class="tick" style="--t:{num(t - base)}"></span>' for t in ticks)
        out.append(f'<div class="row"{tip}><div class="lab">{lab}</div><div class="plot">{"".join(tracks)}{tk}</div></div>')
    if axis:
        ax = "".join(f'<span class="{c}" style="--t:{num(t - base)}">{l}</span>' for t, l, c in axis)
        out.append(f'<div class="row axrow" aria-hidden="true"><div class="lab"></div><div class="axis">{ax}</div></div>')
    out.append("</div>")
    return "".join(out)


def legend(items):
    return '<ul class="legend">' + "".join(f'<li><i class="sw {c}"></i>{t}</li>' for c, t in items) + "</ul>"


def badge(n):
    return f'<span class="bdg" aria-label="Bewegung {n}">{n}</span>'


def ruler(blue=(), tail=(), soft=(), mini=False):
    cells = []
    for i in range(28):
        c = "b" if i in blue else ("t" if i in tail else ("s" if i in soft else ""))
        cells.append(f'<i class="{c}"></i>')
    if mini:
        return f'<span class="ruler mini" aria-hidden="true">{"".join(cells)}</span>'
    nums = "".join(f'<span style="grid-column:{i + 1}">{i}</span>' for i in (0, 18, 21, 27))
    return f'<div class="rulerbox" aria-hidden="true"><div class="ruler">{"".join(cells)}</div><div class="rulernums">{nums}</div></div>'


def dumbbell(rows, lo=-0.2, hi=0.7, ticks=(-0.2, 0.0, 0.2, 0.4, 0.6)):
    """Je Zeile zwei Werte auf einer Skala. Blau, wenn hoechstens 5 % der Zufallswort-Achsen gleich stark sind, sonst grau."""
    def p(v):
        return (v - lo) / (hi - lo)

    out = ['<div class="db">']
    for r in rows:
        a, w = r["all"], r["within"]
        ca, cw = ("" if r["all_ok"] else " off"), ("" if r["within_ok"] else " off")
        l, rr = min(p(a), p(w)), max(p(a), p(w))
        out.append(
            f'<div class="row" data-tip="{esc(r["tip"])}" tabindex="0"><div class="lab">{r["lab"]}</div><div class="dbcell">'
            f'<div class="dbtrack"><span class="zero" style="--p:{num(p(0))}"></span><span class="link" style="--p:{num(l)};--w:{num(rr - l)}"></span>'
            f'<span class="dot hollow{cw}" style="--p:{num(p(w))}"></span><span class="dot solid{ca}" style="--p:{num(p(a))}"></span></div>'
            f'<div class="dbvals"><span><i class="sw dot{ca}"></i><span><b>{r["all_s"]}</b> über alle Fragen · gleich stark: {r["all_ctl"]} der Zufallswort-Achsen</span></span>'
            f'<span><i class="sw ring{cw}"></i><span><b>{r["within_s"]}</b> innerhalb einer Kategorie · gleich stark: {r["within_ctl"]}</span></span></div></div></div>'
        )
    ax = "".join(f'<span style="--p:{num(p(t))}">{str(t).replace(".", ",").replace("-", "−") if t else "0"}</span>' for t in ticks)
    out.append(f'<div class="row axrow" aria-hidden="true"><div class="lab"></div><div class="dbcell"><div class="dbaxis">{ax}</div></div></div></div>')
    return "".join(out)


def table(head, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join((f"<th>{c}</th>" if i == 0 else f"<td>{c}</td>") for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="scroll"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


# ----------------------------------------------------------------------------------------------------------------------
# Diagramm 6.1: Verlustanstieg beim Ausduennen (Linien, x logarithmisch)
# ----------------------------------------------------------------------------------------------------------------------
def sparsity_svg():
    L, W, T, H = 30.0, 270.0, 24.0, 174.0

    def x(p):
        return L + math.log(100.0 / p) / math.log(25.0) * W

    def y(v):
        return T + (12.0 - v) / 12.0 * H

    neu = [(50, 0.02), (25, 0.17), (12.5, 0.70), (7.8, 1.49), (4, 2.93)]
    rot = [(90, 0.02), (75, 0.38), (50, 4.7), (25, 7.6), (12.5, 9.9), (7.8, 10.2)]
    rnd = [(50, 6.9), (25, 10.1), (7.8, 11.8)]
    s = ['<svg class="chart" viewBox="0 0 340 240" role="img" aria-label="Verlustanstieg in nats gegen den behaltenen Anteil der Aktivierungen. '
         'Neuronen-Basis: 50 % plus 0,02, 25 % plus 0,17, 7,8 % plus 1,49, 4 % plus 2,93. Gedrehte Basis: 50 % plus 4,7, 7,8 % plus 10,2. '
         'Zufällige Auswahl: 50 % plus 6,9, 7,8 % plus 11,8.">']
    for v in (0, 3, 6, 9, 12):
        s.append(f'<line class="grid" x1="{L}" y1="{y(v):.1f}" x2="{L + W}" y2="{y(v):.1f}"/>')
        s.append(f'<text class="tk" x="{L - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v}</text>')
    s.append(f'<text class="tk" x="{L - 6}" y="{T - 10}" text-anchor="end">nats</text>')
    s.append(f'<line class="guide" x1="{x(7.8):.1f}" y1="{T}" x2="{x(7.8):.1f}" y2="{T + H}"/>')
    s.append(f'<line class="base" x1="{L}" y1="{y(0):.1f}" x2="{L + W}" y2="{y(0):.1f}"/>')
    for p, lab in ((100, "100 %"), (90, ""), (75, ""), (50, "50 %"), (25, "25 %"), (12.5, "12,5 %"), (7.8, "7,8 %"), (4, "4 %")):
        s.append(f'<line class="base" x1="{x(p):.1f}" y1="{y(0):.1f}" x2="{x(p):.1f}" y2="{y(0) + 4:.1f}"/>')
        if lab:
            s.append(f'<text class="tk" x="{x(p):.1f}" y="{y(0) + 17:.1f}" text-anchor="middle">{lab}</text>')
    s.append(f'<text class="tk" x="{L + W}" y="{y(0) + 35:.1f}" text-anchor="end">behaltener Anteil je Block (log) · weniger →</text>')
    series = ((rnd, "gray thin"), (rot, "gray"), (neu, "blue"))
    for pts, c in series:                                  # erst alle Linien, dann alle Punkte
        full = pts if "thin" in c else [(100, 0.0)] + pts  # zufaellige Auswahl: nur gemessene Punkte verbinden
        s.append(f'<polyline class="ln {c}" fill="none" points="{" ".join(f"{x(p):.1f},{y(v):.1f}" for p, v in full)}"/>')
    for pts, c in series:
        for p, v in pts:
            s.append(f'<circle class="pt {c}" cx="{x(p):.1f}" cy="{y(v):.1f}" r="3.6"/>')
    # Namen der Linien
    s.append(f'<text class="nm" x="{x(25) - 40:.1f}" y="{y(0.17) - 21:.1f}">Neuronen-Basis</text>')
    s.append(f'<text class="nm" x="{x(25) + 2:.1f}" y="{y(7.6) + 19:.1f}">gedrehte Basis</text>')
    s.append(f'<text class="nm" x="{x(7.8) - 14:.1f}" y="{y(11.8) - 7:.1f}" text-anchor="end">zufällige Auswahl</text>')
    # ausgewaehlte Werte
    s.append(f'<text class="vl" x="{x(50) - 7:.1f}" y="{y(4.7) + 4:.1f}" text-anchor="end">+4,7</text>')
    s.append(f'<text class="vl" x="{x(50) - 7:.1f}" y="{y(6.9) + 4:.1f}" text-anchor="end">+6,9</text>')
    s.append(f'<text class="vl" x="{x(50):.1f}" y="{y(0.02) - 8:.1f}" text-anchor="middle">+0,02</text>')
    s.append(f'<text class="vl" x="{x(7.8) + 7:.1f}" y="{y(11.8) + 8:.1f}">+11,8</text>')
    s.append(f'<text class="vl" x="{x(7.8) + 7:.1f}" y="{y(10.2) + 5:.1f}">+10,2</text>')
    s.append(f'<text class="vl" x="{x(7.8) - 7:.1f}" y="{y(1.49) - 7:.1f}" text-anchor="end">+1,49</text>')
    s.append(f'<text class="vl" x="{x(4) + 6:.1f}" y="{y(2.93) + 4:.1f}">+2,93</text>')
    s.append("</svg>")
    return "".join(s)


# ----------------------------------------------------------------------------------------------------------------------
# Skizze 4.2: vier Bewegungen in der Ebene aus Mittel-Differenz und Rest der Trennlinien-Normalen (schematisch, 45 Grad)
# ----------------------------------------------------------------------------------------------------------------------
def schematic_svg():
    ox, oy, u = 46.0, 146.0, 62.0
    r2 = math.sqrt(0.5)
    a = (ox, oy - math.sqrt(2) * u)            # nur fuer die Trennlinie sichtbar, bis zur Schwelle
    b = (ox + r2 * u, oy - r2 * u)              # Normale, Laenge 1
    c = (ox + 1.8 * u, oy)                      # volle Mittel-Differenz, Laenge 1,8
    d = (ox + 1.8 * u, oy + 1.8 * u)            # volle Mittel-Differenz, Anzeige unveraendert
    k = a[1] - a[0]                             # Schwelle: y = x + k; Linie gleicher Anzeige durch A: y = x + (oy - ox)

    def outcome(x, y, p, o):
        return (f'<rect class="f-blue" x="{x}" y="{y - 9}" width="9" height="9" rx="2"/><text class="vl" x="{x + 13}" y="{y}">{p}</text>'
                f'<rect class="f-orange" x="{x + 56}" y="{y - 9}" width="9" height="9" rx="2"/><text class="vl" x="{x + 69}" y="{y}">{o}</text>')

    def bdg(x, y, n):
        return f'<circle class="f-ink" cx="{x}" cy="{y}" r="8"/><text class="bn" x="{x}" y="{y + 3.9}" text-anchor="middle">{n}</text>'

    bottom = 320.0
    s = ['<svg class="chart schem" viewBox="0 0 340 324" role="img" aria-label="Skizze: vier Bewegungen eines A-Zustands. Waagerecht die Richtung der '
         'Mittel-Differenz, auf die das Netz reagiert. Die Schwelle der Trennlinie verläuft schräg dazu. Bewegung 1 (Normale, 44 % der Mittel-Differenz) und Bewegung 2 '
         '(nur für die Trennlinie sichtbar, ohne Anteil der Mittel-Differenz) erreichen die Schwelle. Bewegung 3 (volle Mittel-Differenz) '
         'überschreitet sie. Bewegung 4 trägt die volle Mittel-Differenz und läuft entlang der Linie gleicher Anzeige, parallel zur Schwelle.">',
         '<defs><marker id="ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path class="f-ink" d="M0,0.8 L10,5 L0,9.2 z"/></marker></defs>']
    x_end = bottom - k
    s.append(f'<polygon class="zone" points="8,{8 + k:.1f} 8,8 332,8 332,{bottom:.0f} {x_end:.1f},{bottom:.0f}"/>')
    s.append(f'<line class="lvl" x1="8" y1="{8 + k:.1f}" x2="{x_end:.1f}" y2="{bottom:.0f}"/>')
    s.append(f'<line class="lvl0" x1="8" y1="{oy - ox + 8:.1f}" x2="{ox}" y2="{oy}"/>')
    s.append(f'<line class="ax" x1="14" y1="{oy}" x2="332" y2="{oy}"/><line class="ax" x1="{ox}" y1="262" x2="{ox}" y2="30"/>')
    for t in (a, b, c, d):
        s.append(f'<line class="mv" x1="{ox}" y1="{oy}" x2="{t[0]:.1f}" y2="{t[1]:.1f}" marker-end="url(#ah)"/>')
    s.append(f'<circle class="org" cx="{ox}" cy="{oy}" r="5"/>')
    s.append('<text class="nm" x="24" y="166">A</text>')
    s.append('<text class="rg" x="326" y="24" text-anchor="end">Schwelle der Trennlinie: ab hier „typisches B“</text>')
    s.append('<text class="rg" x="332" y="163" text-anchor="end">Mittel-Differenz A → B:</text><text class="rg" x="332" y="178" text-anchor="end">darauf reagiert das Netz</text>')
    s.append(bdg(60, 39, 2) + '<text class="nm" x="73" y="43">nur für die Trennlinie sichtbar</text>' + outcome(73, 60, "100 %", "6 %"))
    s.append(bdg(104, 84, 1) + '<text class="nm" x="117" y="88">Normale</text>' + outcome(117, 105, "100 %", "12 %"))
    s.append(bdg(172, 116, 3) + '<text class="nm" x="185" y="120">volle Mittel-Differenz</text>' + outcome(185, 137, "100 %", "67 %"))
    s.append(bdg(66, 281, 4) + '<text class="nm" x="79" y="285">volle Mittel-Differenz,</text><text class="nm" x="79" y="300">für die Trennlinie unsichtbar</text>' + outcome(79, 317, "0 %", "68 %"))
    s.append("</svg>")
    return "".join(s)


# ----------------------------------------------------------------------------------------------------------------------
# Inhalte der Diagramme (Zahlen aus results/ju_space/README.md)
# ----------------------------------------------------------------------------------------------------------------------
FIG_1A = bars(
    [
        dict(lab="ganzer Zustand", sub="50 Hauptachsen", bars=[dict(v=86, disp="86 %", cls="ref")], tip="Ganzer Zustand (50 Hauptachsen): 86 %"),
        dict(lab="U-Space, 4 Achsen", bars=[dict(v=38.7, disp="38,7 %", cls="blue")], tip="U-Space (4 Achsen): 38,7 %"),
        dict(lab="4 Achsen aus Zufallswörtern", sub="gleiches Rezept", bars=[dict(v=42, ext=44, rng=(36, 49), disp="42–44 %", cls="gray")],
             tip="4 Achsen nach demselben Rezept aus Zufallswörtern: 42–44 % (zwei Wortvorräte)\n5.–95. Perzentil beider Vorräte: 36–49 %"),
        dict(lab="4 zufällige Richtungen", sub="orthogonal", bars=[dict(v=38, rng=(32, 44), disp="38 %", cls="gray")],
             tip="4 zufällige orthogonale Richtungen: 38 %\n5.–95. Perzentil: 32–44 %"),
    ],
    top=100, vw="5.4rem", ticks=(5.6,), axis=((0, "0", "first"), (50, "50 %", ""), (100, "100 %", "")),
)

FIG_2 = dumbbell(
    [
        dict(lab="Mehrdeutigkeit", all=0.41, within=0.15, all_s="0,41", within_s="0,15", all_ctl="2 %", within_ctl="15 %", all_ok=True, within_ok=False,
             tip="Mehrdeutigkeit, Block 18 / 21\nalle Fragen: 0,41 / 0,32\ninnerhalb einer Kategorie: 0,15 / 0,07\nZufallswort-Achsen gleich stark: 2 % / 5 % (innerhalb 15 % / 44 %)"),
        dict(lab="Unvollständigkeit", all=-0.12, within=0.06, all_s="−0,12", within_s="0,06", all_ctl="51 %", within_ctl="55 %", all_ok=False, within_ok=False,
             tip="Unvollständigkeit, Block 18 / 21\nalle Fragen: −0,12 / −0,19\ninnerhalb einer Kategorie: 0,06 / 0,02\nZufallswort-Achsen gleich stark: 51 % / 39 % (innerhalb 55 % / 88 %)"),
        dict(lab="<strong>widersprüchliche Belege</strong>", all=0.61, within=0.32, all_s="0,61", within_s="0,32", all_ctl="2 %", within_ctl="4 %", all_ok=True, within_ok=True,
             tip="Widersprüchliche Belege, Block 18 / 21\nalle Fragen: 0,61 / 0,54\ninnerhalb einer Kategorie: 0,32 / 0,40\nZufallswort-Achsen gleich stark: 2 % / 3 % (innerhalb 4 % / unter 1 %)"),
        dict(lab="allgemeine Unsicherheit", all=0.45, within=0.26, all_s="0,45", within_s="0,26", all_ctl="18 %", within_ctl="1 %", all_ok=False, within_ok=True,
             tip="Allgemeine Unsicherheit, Block 18 / 21\nalle Fragen: 0,45 / 0,29\ninnerhalb einer Kategorie: 0,26 / 0,22\nZufallswort-Achsen gleich stark: 18 % / 39 % (innerhalb 1 % / 1 %)"),
    ]
)

FIG_3 = bars(
    [
        dict(lab="direkte Differenzrichtung", sub="obere Vergleichsmarke", bars=[dict(v=0.78, disp="0,78", cls="ref")], tip="Direkte Differenzrichtung: 0,78 (Block 21: 0,76)"),
        dict(lab="<strong>U++</strong>", sub="Anker = eigene Wörter des Modells", bars=[dict(v=0.75, disp="0,75", cls="blue")], tip="U++: 0,75 (Block 21: 0,74)"),
        dict(lab="eine gemeinsame Richtung", sub="Mittel der direkten Richtungen der anderen 15 Verhaltensweisen", bars=[dict(v=0.72, disp="0,72", cls="blue")], tip="Eine gemeinsame Richtung: Mittel der direkten Differenzrichtungen der anderen 15 Verhaltensweisen\n0,72 (Block 21: 0,70)"),
        dict(lab="generischer U-Space", bars=[dict(v=0.64, disp="0,64", cls="blue")], tip="Generischer U-Space: 0,64 (Block 21: 0,61)"),
        dict(lab="15 Zufallswörter", sub="gleiches Rezept", bars=[dict(v=0.52, disp="0,52", cls="gray")], tip="Dasselbe Rezept mit 15 Zufallswörtern: 0,52 (Block 21: 0,52)"),
        dict(lab="nur die Antwortlänge", bars=[dict(v=0.51, disp="0,51", cls="gray")], tip="Nur die Antwortlänge: 0,51"),
    ],
    base=0.5, top=1.0, vw="3.2rem", axis=((0.5, "0,5 Zufall", "first"), (0.75, "0,75", ""), (1.0, "1,0", "")),
)

FIG_4A = bars(
    [
        dict(lab="hartes SVM", sub="2.048 Dimensionen", bars=[dict(v=98.9, disp="98,9 %", cls="blue")], tip="Hartes SVM: 98,9 %"),
        dict(lab="Mittel-Differenz", bars=[dict(v=96.2, disp="96,2 %", cls="blue")], tip="Mittel-Differenz (Differenz der Kategorienmittel): 96,2 %"),
        dict(lab="20 SAE-Einheiten", bars=[dict(v=94.7, disp="94,7 %", cls="blue")], tip="20 SAE-Einheiten: 94,7 %\n5 Einheiten: 94,0 %"),
        dict(lab="beste einzelne SAE-Einheit", bars=[dict(v=90.4, disp="90,4 %", cls="blue")], tip="Beste einzelne SAE-Einheit: 90,4 %"),
        dict(lab="U-Space, 4 Achsen", bars=[dict(v=88.3, disp="88,3 %", cls="blue")], tip="U-Space: 88,3 %"),
        dict(lab="4 Zufallsrichtungen", bars=[dict(v=84.4, disp="84,4 %", cls="gray")], tip="4 Zufallsrichtungen: 84,4 %"),
        dict(lab="hartes SVM, gewürfelte Etiketten", sub="im Training: 100 %", bars=[dict(v=50.9, disp="50,9 %", cls="gray")],
             tip="Dasselbe SVM auf gewürfelten Etiketten\nTraining: 100 %\nPrüfung: 50,9 %"),
    ],
    base=50, top=100, vw="4.4rem", axis=((50, "50 % Zufall", "first"), (75, "75 %", ""), (100, "100 %", "")),
)

P = lambda t, o: [dict(v=t, disp=f"{t} %", cls="blue"), dict(v=o, disp=f"{o} %", cls="orange")]  # noqa: E731
FIG_4C = bars(
    [
        dict(lab="unverändert", bars=P(0, 4), tip="Unverändert\nTrennlinie liest B: 0 % · Ausgabe liest B: 4 %\nzweite Trennlinie: 2 % · untypisch: 18 %"),
        dict(lab=badge(1) + "Normale der Trennlinie", sub="λ = 0, Länge 1,0", bars=P(100, 12),
             tip="Normale der Trennlinie (λ = 0)\nLänge 1,0 · Anteil der Mittel-Differenz 0,44\nTrennlinie liest B: 100 % · Ausgabe liest B: 12 %\nzweite Trennlinie: 57 % · untypisch: 4 %\nAusgabe je Paar: 6 / 3 / 7 / 31 %"),
        dict(lab=badge(2) + "nur für die Trennlinie sichtbar", sub="Länge 1,5", bars=P(100, 6),
             tip="Nur für die Trennlinie sichtbar\nLänge 1,5 · Anteil der Mittel-Differenz 0,00\nTrennlinie liest B: 100 % · Ausgabe liest B: 6 %\nzweite Trennlinie: 5 % · untypisch: 26 %\nAusgabe je Paar: 3 / 0 / 4 / 16 %"),
        dict(lab=badge(3) + "volle Mittel-Differenz", sub="Länge 1,8", bars=P(100, 67),
             tip="Volle Mittel-Differenz\nLänge 1,8 · Anteil der Mittel-Differenz 1,00\nTrennlinie liest B: 100 % · Ausgabe liest B: 67 %\nzweite Trennlinie: 99 % · untypisch: 17 %"),
        dict(lab=badge(4) + "volle Mittel-Differenz, für die Trennlinie unsichtbar", sub="Länge 2,5", bars=P(0, 68),
             tip="Volle Mittel-Differenz, für die Trennlinie unsichtbar\nLänge 2,5 · Anteil der Mittel-Differenz 1,00\nTrennlinie liest B: 0 % · Ausgabe liest B: 68 %\nzweite Trennlinie: 93 % · untypisch: 28 %\nAusgabe je Paar: 45 / 97 / 57 / 75 %"),
        dict(lab="echte B-Fragen", sub="zurückgehalten", bars=P(95, 73),
             tip="Echte zurückgehaltene B-Fragen\nTrennlinie liest B: 95 % · Ausgabe liest B: 73 %\nzweite Trennlinie: 96 % · untypisch: 6 %\nAusgabe je Paar: 67 / 89 / 53 / 84 %"),
    ],
    top=100, vw="4rem", axis=((0, "0", "first"), (50, "50 %", ""), (100, "100 %", "")), cls="widelab",
)

TABLE_42 = table(
    ["Zustand", "Länge", "Anteil der Mittel-Differenz", "Trennlinie liest B", "zweite, unabhängige Trennlinie", "Ausgabe liest B", "untypisch"],
    [
        ["unverändert", "0", "0", "0 %", "2 %", "4 %", "18 %"],
        ["Zufallsrichtung", "1,0", "0", "0 %", "2 %", "6 %", "43 %"],
        ["nur für die Trennlinie sichtbar", "1,5", "0,00", "100 %", "5 %", "6 %", "26 %"],
        ["Normale der Trennlinie (λ = 0)", "1,0", "0,44", "100 %", "57 %", "12 %", "4 %"],
        ["λ = 1", "1,2", "0,60", "100 %", "85 %", "26 %", "59 %"],
        ["λ = 5", "1,9", "0,80", "100 %", "100 %", "57 %", "99 %"],
        ["λ = ∞ (Schwerpunkt von B)", "2,5", "0,96", "100 %", "100 %", "94 %", "100 %"],
        ["Verschiebung um die Mittel-Differenz bis zur selben Anzeige", "1,6", "0,91", "100 %", "98 %", "61 %", "17 %"],
        ["volle Mittel-Differenz", "1,8", "1,00", "100 %", "99 %", "67 %", "17 %"],
        ["volle Mittel-Differenz, für die Trennlinie unsichtbar", "2,5", "1,00", "0 %", "93 %", "68 %", "28 %"],
        ["echter B-Zustand eingesetzt", "3,4", "0,98", "98 %", "97 %", "79 %", "7 %"],
        ["echte zurückgehaltene B-Fragen", "–", "–", "95 %", "96 %", "73 %", "6 %"],
    ],
)

L = lambda o, u: [dict(v=o, disp=f"{o} %", cls="orange"), dict(v=u, disp=f"{u} %", cls="hollow")]  # noqa: E731
FIG_4D = bars(
    [
        dict(grp="Angriff mit Dichte-Term"),
        dict(lab="λ = 0", sub="Anteil der Mittel-Differenz 0,44", bars=L(12, 4), tip="λ = 0 (Normale der Trennlinie)\nLänge 1,0 · Anteil der Mittel-Differenz 0,44\nAusgabe liest B: 12 % · untypisch: 4 %"),
        dict(lab="λ = 1", sub="Anteil 0,60", bars=L(26, 59), tip="λ = 1\nLänge 1,2 · Anteil der Mittel-Differenz 0,60\nAusgabe liest B: 26 % · untypisch: 59 %"),
        dict(lab="λ = 5", sub="Anteil 0,80", bars=L(57, 99), tip="λ = 5\nLänge 1,9 · Anteil der Mittel-Differenz 0,80\nAusgabe liest B: 57 % · untypisch: 99 %"),
        dict(lab="λ = ∞", sub="Schwerpunkt von B, Anteil 0,96", bars=L(94, 100), tip="λ = ∞ (Schwerpunkt von B)\nLänge 2,5 · Anteil der Mittel-Differenz 0,96\nAusgabe liest B: 94 % · untypisch: 100 %"),
        dict(grp="Zum Vergleich"),
        dict(lab="volle Mittel-Differenz", sub="Anteil 1,00", bars=L(67, 17), tip="Volle Mittel-Differenz\nLänge 1,8 · Anteil 1,00\nAusgabe liest B: 67 % · untypisch: 17 %"),
        dict(lab="echte B-Fragen", bars=L(73, 6), tip="Echte zurückgehaltene B-Fragen\nAusgabe liest B: 73 % · untypisch: 6 %"),
        dict(lab="unverändert", bars=L(4, 18), tip="Unverändert\nAusgabe liest B: 4 % · untypisch: 18 %"),
    ],
    top=100, vw="4rem", axis=((0, "0", "first"), (50, "50 %", ""), (100, "100 %", "")),
)

FIG_4E = bars(
    [
        dict(lab="Normale der Trennlinie", bars=P(100, 12), tip="Normale der Trennlinie, Länge 1\nTrennlinie liest B: 100 % · Ausgabe liest B: 12 %"),
        dict(lab="Mittel-Differenz", bars=P(92, 28), tip="Mittel-Differenz, Länge 1\nTrennlinie liest B: 92 % · Ausgabe liest B: 28 %"),
        dict(lab="Richtung zum Schwerpunkt von&nbsp;B", bars=P(40, 17), tip="Richtung zum Schwerpunkt von B, Länge 1\nTrennlinie liest B: 40 % · Ausgabe liest B: 17 %"),
        dict(lab="20 SAE-Einheiten", sub="Richtung aus den 20 am besten trennenden", bars=P(16, 27), tip="Richtung aus den 20 am besten trennenden SAE-Einheiten, Länge 1\nTrennlinie liest B: 16 % · Ausgabe liest B: 27 %"),
        dict(lab="Lens-Gradient", sub="steilster Anstieg unter der J-Lens", bars=P(6, 14), tip="Lens-Gradient (steilster Anstieg unter der J-Lens), Länge 1\nTrennlinie liest B: 6 % · Ausgabe liest B: 14 %"),
        dict(lab="Zufallsrichtung", bars=P(0, 6), tip="Zufallsrichtung, Länge 1\nTrennlinie liest B: 0 % · Ausgabe liest B: 6 %"),
    ],
    top=100, vw="4rem", axis=((0, "0", "first"), (50, "50 %", ""), (100, "100 %", "")),
)

B1 = lambda v, cls, s=None, rng=None: [dict(v=v, disp=s or f"{v} %", cls=cls, **({"rng": rng} if rng else {}))]  # noqa: E731
FIG_5 = bars(
    [
        dict(grp="Richtungen im Zustand, Block 21"),
        dict(lab="Normale eines harten SVM", sub="4 Einheiten", bars=B1(94, "blue", rng=(81.2, 100)), tip="Normale eines harten SVM (4 Einheiten): 94 % (95-%-Intervall des Mittels: 81–100 %)"),
        dict(lab="Mittel-Differenz", sub="4 Einheiten", bars=B1(88, "blue", rng=(71.9, 100)), tip="Mittel-Differenz (4 Einheiten): 88 % (95-%-Intervall des Mittels: 72–100 %)"),
        dict(lab="U-Achsen", sub="4 Einheiten", bars=B1(81, "blue", rng=(75, 87.5)), tip="U-Achsen (4 Einheiten): 81 % (95-%-Intervall des Mittels: 75–88 %)"),
        dict(lab="SAE-Einheiten", sub="14 Einheiten", bars=B1(77, "blue", rng=(69.6, 83.0)), tip="SAE-Einheiten (14 Einheiten): 77 % (95-%-Intervall des Mittels: 70–83 %)"),
        dict(lab="Richtungen aus Zufallswörtern", sub="4 Einheiten, Kontrolle", bars=B1(75, "gray", rng=(62.5, 87.5)), tip="Richtungen aus Zufallswörtern (4 Einheiten): 75 % (95-%-Intervall des Mittels: 63–88 %)"),
        dict(lab="Zufallsrichtungen", sub="4 Einheiten, Kontrolle", bars=B1(69, "gray", rng=(62.5, 75)), tip="Zufallsrichtungen (4 Einheiten): 69 % (95-%-Intervall des Mittels: 63–75 %)"),
        dict(grp="Neuronen"),
        dict(lab="gezielt gewählte Neuronen", sub="12 Neuronen", bars=B1(83, "blue", rng=(75, 90.6)), tip="Gezielt gewählte Neuronen (12, führende Träger der Kategoriendifferenzen): 83 % (95-%-Intervall des Mittels: 75–91 %)"),
        dict(lab="zufällig gezogene Neuronen", sub="6 Neuronen, Blöcke 10–21", bars=B1(73, "blue", rng=(66.7, 79.2)), tip="Zufällig gezogene Neuronen der Blöcke 10–21 (6): 73 % (95-%-Intervall des Mittels: 67–79 %)"),
        dict(lab="zufällige Mischungen von Neuronen", sub="6 Mischungen, Kontrolle", bars=B1(83, "gray", rng=(72.9, 93.8)), tip="Zufällige Mischungen der Neuronen eines Blocks (6): 83 % (95-%-Intervall des Mittels: 73–94 %)"),
    ],
    base=50, top=100, vw="3.4rem", axis=((50, "50 % Zufall", "first"), (75, "75 %", ""), (100, "100 %", "")),
)

FIG_6B = bars(
    [
        dict(grp="Führende Neuronen (von 135.168 in den Blöcken 0–21)"),
        dict(lab="20", sub="Anteil der Mittel-Differenz 0,19", bars=B1(9, "orange"), tip="20 führende Neuronen\nAnteil der Mittel-Differenz 0,19 · Ausgabe liest B: 9 %"),
        dict(lab="100", sub="Anteil 0,37", bars=B1(28, "orange"), tip="100 führende Neuronen\nAnteil 0,37 · Ausgabe liest B: 28 %"),
        dict(lab="300", sub="Anteil 0,53", bars=B1(53, "orange"), tip="300 führende Neuronen\nAnteil 0,53 · Ausgabe liest B: 53 %"),
        dict(lab="1.000", sub="Anteil 0,68", bars=B1(69, "orange"), tip="1.000 führende Neuronen\nAnteil 0,68 · Ausgabe liest B: 69 %"),
        dict(lab="3.000", sub="Anteil 0,75", bars=B1(68, "orange"), tip="3.000 führende Neuronen\nAnteil 0,75 · Ausgabe liest B: 68 %"),
        dict(grp="Führende SAE-Einheiten (von 16.384, gleiches Auswahlkriterium)"),
        dict(lab="20", sub="Anteil 0,30", bars=B1(30, "orange"), tip="20 führende SAE-Einheiten\nAnteil 0,30 · Ausgabe liest B: 30 %"),
        dict(lab="100", sub="Anteil 0,51", bars=B1(71, "orange"), tip="100 führende SAE-Einheiten\nAnteil 0,51 · Ausgabe liest B: 71 %"),
        dict(lab="300", sub="Anteil 0,59", bars=B1(77, "orange"), tip="300 führende SAE-Einheiten\nAnteil 0,59 · Ausgabe liest B: 77 %"),
        dict(grp="Kontrollen und Vergleich"),
        dict(lab="20 bis 3.000 zufällige Neuronen", sub="Anteil 0,00–0,03", bars=B1(4, "gray"), tip="20 bis 3.000 zufällige Neuronen\nAnteil 0,00–0,03 · Ausgabe liest B: 4 %"),
        dict(lab="unverändert", bars=B1(4, "gray"), tip="Unverändert: Ausgabe liest B in 4 %"),
        dict(lab="der von der Attention geschriebene Teil", sub="Anteil 0,21", bars=B1(4, "orange"), tip="Der von der Attention geschriebene Teil der Differenz\nAnteil 0,21 · Ausgabe liest B: 4 %"),
        dict(lab="alle Neuronen", sub="der von MLPs geschriebene Teil, Anteil 0,79", bars=B1(75, "orange"), tip="Alle Neuronen (der von MLPs geschriebene Teil)\nAnteil 0,79 · Ausgabe liest B: 75 %"),
        dict(lab="volle Mittel-Differenz", sub="Anteil 1,00", bars=B1(67, "ref"), tip="Volle Mittel-Differenz (aus Abschnitt 4)\nAnteil 1,00 · Ausgabe liest B: 67 %"),
        dict(lab="echte B-Fragen", bars=B1(73, "ref"), tip="Echte B-Fragen: Ausgabe liest B in 73 %"),
    ],
    top=100, vw="3.4rem", ticks=(67,), axis=((0, "0", "first"), (67, "67 %", ""), (100, "100 %", "")),
)

D = lambda t, f, fs: [dict(v=t, disp=f"{t} %", cls="blue"), dict(v=100 * f, disp=fs, cls="orange")]  # noqa: E731
FIG_6C = bars(
    [
        dict(lab="Zufallsrichtung ähnlicher Länge", bars=D(0, 0.03, "0,03"), tip="Zufallsrichtung ähnlicher Länge (2×)\nTrennlinie liest B: 0 %\nAnteil am echten Unterschied: 0,03 · Kosinus 0,02"),
        dict(lab=badge(2) + "nur für die Trennlinie sichtbar", bars=D(100, 0.06, "0,06"),
             tip="Nur für die Trennlinie sichtbar, bis sie „typisches B“ meldet\nTrennlinie liest B: 100 %\nAnteil am echten Unterschied: 0,06 · Kosinus 0,04"),
        dict(lab=badge(1) + "Normale der Trennlinie", bars=D(100, 0.34, "0,34"), tip="Normale der Trennlinie (λ = 0)\nTrennlinie liest B: 100 %\nAnteil am echten Unterschied: 0,34 · Kosinus 0,37"),
        dict(lab=badge(4) + "volle Mittel-Differenz, für die Trennlinie unsichtbar", bars=D(0, 0.84, "0,84"),
             tip="Volle Mittel-Differenz, für die Trennlinie unsichtbar\nTrennlinie liest B: 0 %\nAnteil am echten Unterschied: 0,84 · Kosinus 0,34\nin sieben Paaren 0,77–0,95"),
        dict(lab=badge(3) + "volle Mittel-Differenz", bars=D(100, 0.91, "0,91"), tip="Volle Mittel-Differenz\nTrennlinie liest B: 100 %\nAnteil am echten Unterschied: 0,91 · Kosinus 0,57"),
        dict(lab="echter B-Zustand eingesetzt", bars=D(98, 0.94, "0,94"), tip="Echter B-Zustand eingesetzt\nTrennlinie liest B: 98 %\nAnteil am echten Unterschied: 0,94 (Mittel) · Kosinus 0,37"),
    ],
    top=100, vw="4rem", axis=((0, "0", "first"), (50, "50 % · 0,5", ""), (100, "100 % · 1,0", "")), cls="widelab",
)

TABLE_61 = table(
    ["behalten", "Neuronen-Basis", "zufällig gedrehte Basis", "zufällige Auswahl"],
    [["90 %", "–", "+0,02", "–"], ["75 %", "–", "+0,38", "–"], ["50 %", "+0,02", "+4,7", "+6,9"], ["25 %", "+0,17", "+7,6", "+10,1"],
     ["12,5 %", "+0,70", "+9,9", "–"], ["7,8 %", "+1,49", "+10,2", "+11,8"], ["4 %", "+2,93", "–", "–"]],
    cls="narrow",
)

# ----------------------------------------------------------------------------------------------------------------------
# Seite
# ----------------------------------------------------------------------------------------------------------------------
CSS = r"""
/* Aufbau: eine Spalte nummerierter Fragen; nur Diagramme stehen auf Karten; ein Lineal aus 28 Zellen zeigt, an welchem Block gemessen wurde */
:root{
  --bg:#f3f5f8; --surface:#ffffff; --ink:#10151b; --ink-2:#454e59; --muted:#636d79;
  --line:#dde2e8; --line-2:#c3cad3; --chip:#e8ecf1;
  --blue:#2a78d6; --blue-soft:#e3eefb; --orange:#eb6834; --orange-soft:#fbe3d8; --gray:#8a94a0; --ref:#3b444e;
  --font-text:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --font-mono:"IBM Plex Mono",ui-monospace,"SFMono-Regular",Menlo,Consolas,"Liberation Mono",monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#0e1216; --surface:#161b21; --ink:#e9eef3; --ink-2:#b4bec9; --muted:#8f9aa6;
    --line:#252c34; --line-2:#3a444f; --chip:#222a33;
    --blue:#3987e5; --blue-soft:#14263c; --orange:#d95926; --orange-soft:#3b2014; --gray:#6f7985; --ref:#c6ced7;
    color-scheme:dark;
  }
}
:root[data-theme="dark"]{
  --bg:#0e1216; --surface:#161b21; --ink:#e9eef3; --ink-2:#b4bec9; --muted:#8f9aa6;
  --line:#252c34; --line-2:#3a444f; --chip:#222a33;
  --blue:#3987e5; --blue-soft:#14263c; --orange:#d95926; --orange-soft:#3b2014; --gray:#6f7985; --ref:#c6ced7;
  color-scheme:dark;
}
*,*::before,*::after{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:400 16px/1.55 var(--font-text);-webkit-text-size-adjust:100%;text-size-adjust:100%}
.page{max-width:1120px;margin:0 auto;padding-block:28px 64px;padding-left:max(16px, env(safe-area-inset-left, 0px));padding-right:max(16px, env(safe-area-inset-right, 0px));display:flex;flex-direction:column;gap:64px;overflow-wrap:break-word}
p,ul,ol,h1,h2,h3,h4,figure,dl,dd{margin:0}
ul,ol{padding:0;list-style:none}
strong,b{font-weight:600}
a{color:inherit}
:focus-visible{outline:2px solid var(--blue);outline-offset:3px;border-radius:4px}

/* Kopf */
.head{display:grid;gap:28px}
.head-text{display:grid;gap:16px;align-content:start}
.eyebrow{font:500 .75rem/1.5 var(--font-mono);letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
h1{font:600 clamp(1.8rem,1.15rem + 3vw,2.9rem)/1.08 var(--font-text);letter-spacing:-.018em;text-wrap:balance}
.lead{font-size:clamp(1.05rem,1rem + .35vw,1.22rem);line-height:1.5;color:var(--ink-2);max-width:60ch}
.lead strong{color:var(--ink)}
.key{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:.8125rem;color:var(--ink-2)}
.key li{display:flex;align-items:center;gap:7px}
.sw{display:inline-block;flex:none;width:11px;height:11px;border-radius:3px;background:var(--blue)}
.sw.orange{background:var(--orange)} .sw.gray{background:var(--gray)} .sw.ref{background:var(--ref)}
.sw.ring{background:var(--surface);border:2.5px solid var(--blue);border-radius:50%}
.sw.hollow{background:transparent;border:2px solid var(--ink-2)}
.sw.graydot{background:var(--gray);border-radius:50%}
.sw.grayring{background:var(--surface);border:2.5px solid var(--gray);border-radius:50%}
.sw.dot.off{background:var(--gray)} .sw.ring.off{border-color:var(--gray)}
.sw.dot{border-radius:50%}
.sw.whisker{width:16px;height:2px;border-radius:0;background:var(--ink)}
.sw.vtick{width:2px;height:9px;border-radius:0;background:var(--ink-2);margin-inline:4px}
@media (min-width:900px){.head{grid-template-columns:minmax(0,1.15fr) minmax(0,.85fr);gap:40px;align-items:start}.key{display:grid;grid-template-columns:repeat(2,max-content);gap:6px 22px}}

/* Messstelle */
.site h3{font:600 1rem/1.3 var(--font-text)}
.rulerbox{display:grid;gap:4px}
.ruler{display:grid;grid-template-columns:repeat(28,minmax(0,1fr));gap:2px}
.ruler i{display:block;height:20px;border-radius:2px;background:var(--line)}
.ruler i.b{background:var(--blue)} .ruler i.t{background:var(--orange)} .ruler i.s{background:var(--line-2)}
.rulernums{display:grid;grid-template-columns:repeat(28,minmax(0,1fr));gap:2px;font:500 .6875rem/1 var(--font-mono);color:var(--muted)}
.rulernums span{justify-self:center;white-space:nowrap}
.ruler.mini{display:inline-grid;grid-template-columns:repeat(28,4px);gap:1px;vertical-align:middle}
.ruler.mini i{height:11px;border-radius:1px}
.sitekey{display:grid;gap:6px;font-size:.875rem;color:var(--ink-2)}
.sitekey li{display:grid;grid-template-columns:auto minmax(0,1fr);gap:8px;align-items:baseline}
.sitekey .sw{transform:translateY(1px)}
.facts-dl{display:grid;gap:6px;font-size:.875rem}
.facts-dl div{display:grid;grid-template-columns:4.6em minmax(0,1fr);gap:10px}
.facts-dl dt{color:var(--muted)}
.caveat{font-size:.875rem;line-height:1.5;color:var(--ink);background:var(--chip);border-radius:8px;padding:10px 12px}

/* Kurzfassung */
.sumwrap{display:grid;gap:14px}
.sumwrap h2{font:500 .75rem/1.5 var(--font-mono);letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,310px),1fr));gap:0 36px}
.summary li{border-top:1px solid var(--line-2);display:grid}
.summary a{display:grid;grid-template-columns:1.4em minmax(0,1fr);gap:2px 8px;padding:14px 0 20px;text-decoration:none;align-content:start}
.summary .n{font:500 .8125rem/1.9 var(--font-mono);color:var(--muted)}
.summary .topic{font-size:.8125rem;color:var(--muted)}
.summary .big{grid-column:2;font:600 1.65rem/1.15 var(--font-text);letter-spacing:-.012em}
.summary .txt{grid-column:2;font-size:.9375rem;line-height:1.5;color:var(--ink-2);margin-top:4px}
.summary a:hover .big{text-decoration:underline;text-decoration-thickness:2px;text-underline-offset:5px}

.gloss{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,236px),1fr));gap:16px 36px;font-size:.875rem;line-height:1.5}
.gloss div{display:grid;gap:2px;align-content:start}
.gloss dt{font-weight:600}
.gloss dd{color:var(--ink-2)}

/* Abschnitte */
.sec{display:grid;gap:22px;scroll-margin-top:16px}
.sec-head{display:grid;gap:10px;max-width:74ch}
.sec-tag{display:flex;flex-wrap:wrap;align-items:center;gap:6px 12px;font:500 .75rem/1.5 var(--font-mono);letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
.sec-tag .num{color:var(--bg);background:var(--ink);border-radius:4px;padding:0 7px;font-weight:600}
.sec-tag .loctxt{text-transform:none;letter-spacing:0}
h2.q{font:600 clamp(1.4rem,1.05rem + 1.4vw,1.95rem)/1.16 var(--font-text);letter-spacing:-.012em;text-wrap:balance}
.answer{font-size:1.0625rem;line-height:1.55}
.setup{font-size:.9375rem;line-height:1.55;color:var(--ink-2)}
div.setup{display:grid;gap:10px}
.defs{display:grid;gap:6px}
.defs li{padding-left:14px;position:relative}
.defs li::before{content:"";position:absolute;left:0;top:.62em;width:5px;height:5px;border-radius:50%;background:var(--muted)}
.defs strong{color:var(--ink)}
.figs{display:grid;grid-template-columns:minmax(0,1fr);gap:18px;align-items:stretch}
.stack{display:grid;gap:18px;align-content:start;min-width:0}
@media (min-width:900px){
  .figs{grid-template-columns:repeat(12,minmax(0,1fr));gap:20px}
  .stack{gap:20px}
  .s12{grid-column:span 12} .s7{grid-column:span 7} .s6{grid-column:span 6} .s5{grid-column:span 5} .s4{grid-column:span 4} .s8{grid-column:span 8}
}

/* Karten fuer Diagramme */
.fig{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:16px 14px 14px;display:flex;flex-direction:column;gap:14px;min-width:0;container-type:inline-size}
@media (min-width:600px){.fig{padding:20px 20px 18px}}
.fig figcaption{display:grid;gap:5px}
.fig h3{font:600 1.03rem/1.3 var(--font-text);text-wrap:balance}
.sub{font-size:.8125rem;line-height:1.45;color:var(--muted)}
.note{font-size:.8125rem;line-height:1.5;color:var(--ink-2)}
.note + .note{margin-top:-6px}
.src{font:400 .6875rem/1.4 var(--font-mono);color:var(--muted)}
.fig .src{margin-top:auto}
.nw{white-space:nowrap}
.swipe{display:none;font-size:.75rem;color:var(--muted);margin-top:8px}
@media (max-width:700px){.swipe{display:block}}
.legend{display:flex;flex-wrap:wrap;gap:5px 16px;font-size:.8125rem;color:var(--ink-2)}
.legend li{display:flex;align-items:center;gap:7px}

/* Balken */
.bars{display:grid;gap:9px}
.row{display:grid;gap:3px;min-width:0;border-radius:6px}
.row[data-tip]{cursor:default}
.lab{font-size:.875rem;line-height:1.3;min-width:0}
.lab small{display:block;font-size:.75rem;line-height:1.35;color:var(--muted)}
.grp{font:500 .6875rem/1.4 var(--font-mono);letter-spacing:.05em;text-transform:uppercase;color:var(--muted);padding-top:6px}
.grp:first-child{padding-top:0}
.plot{position:relative;display:grid;gap:2px;border-left:1px solid var(--line-2);min-width:0;padding-block:1px}
.track{display:flex;align-items:center;gap:7px;height:15px;min-width:0}
.bar{display:block;flex:none;height:13px;min-width:2px;border-radius:0 4px 4px 0;background:var(--blue);width:calc((100% - var(--vw)) * var(--v) / var(--m))}
.wrap{position:relative;display:block;flex:none;height:13px;width:calc((100% - var(--vw)) * var(--v) / var(--m))}
.wrap .bar{position:absolute;left:0;top:0;height:100%}
.ext{position:absolute;top:0;height:100%;border-radius:0 4px 4px 0;opacity:.45;background:var(--gray)}
.wrap .bar:has(+ .ext){border-radius:0}
.whisk{position:absolute;top:50%;height:2px;margin-top:-1px;background:var(--ink)}
.whisk::before,.whisk::after{content:"";position:absolute;top:-4px;width:2px;height:10px;background:var(--ink)}
.whisk::before{left:0} .whisk::after{right:0}
.bar.orange,.ext.orange{background:var(--orange)} .bar.gray,.ext.gray{background:var(--gray)} .bar.ref{background:var(--ref)}
.bar.hollow{background:transparent;border:2px solid var(--ink-2);border-left:0;min-width:3px}
.val{position:relative;z-index:1;font:600 .8125rem/1 var(--font-text);font-variant-numeric:tabular-nums;white-space:nowrap;background:var(--surface);padding:1px 2px;margin-inline:-2px}
.tick{position:absolute;top:-6px;height:5px;width:2px;margin-left:-1px;background:var(--ink-2);left:calc((100% - var(--vw)) * var(--t) / var(--m));pointer-events:none}
.axis{position:relative;height:1.5em;border-left:1px solid transparent;font-size:.6875rem;color:var(--muted);font-variant-numeric:tabular-nums}
.axis span{position:absolute;top:1px;left:calc((100% - var(--vw)) * var(--t) / var(--m));transform:translateX(-50%);white-space:nowrap}
.axis span.first{transform:none}
.axis span.chance{transform:translateX(-12%);color:var(--ink-2)}
.axrow{gap:0}
.row[data-tip]:hover .bar,.row[data-tip]:focus-visible .bar{filter:brightness(1.12) saturate(1.05)}
.bdg{display:inline-grid;place-items:center;width:1.25em;height:1.25em;margin-right:.45em;border-radius:50%;background:var(--ink);color:var(--surface);font:600 .72em/1 var(--font-text);vertical-align:.08em}
@container (min-width:470px){
  .row{grid-template-columns:minmax(0,12.5em) minmax(0,1fr);gap:12px;align-items:center}
  .widelab .row{grid-template-columns:minmax(0,15.5em) minmax(0,1fr)}
  .grp{padding-top:8px}
}

/* Hantel-Diagramm */
.db{display:grid;gap:14px}
.dbcell{display:grid;gap:5px;min-width:0}
.dbtrack{position:relative;height:18px;border-radius:9px;background:linear-gradient(var(--line),var(--line)) center/100% 1px no-repeat}
.dbtrack .zero{position:absolute;top:-2px;bottom:-2px;width:1px;background:var(--ink-2);left:calc(9px + (100% - 18px) * var(--p))}
.dbtrack .link{position:absolute;top:50%;height:3px;margin-top:-1.5px;background:var(--line-2);left:calc(9px + (100% - 18px) * var(--p));width:calc((100% - 18px) * var(--w))}
.dbtrack .dot{position:absolute;top:50%;width:14px;height:14px;margin:-7px 0 0 -7px;border-radius:50%;left:calc(9px + (100% - 18px) * var(--p));box-shadow:0 0 0 2px var(--surface)}
.dbtrack .dot.solid{background:var(--blue)}
.dbtrack .dot.hollow{background:var(--surface);border:3px solid var(--blue)}
.dbtrack .dot.solid.off{background:var(--gray)}
.dbtrack .dot.hollow.off{border-color:var(--gray)}
.dbvals{display:grid;gap:3px;font-size:.8125rem;line-height:1.4;color:var(--ink-2)}
.dbvals > span{display:flex;gap:7px;align-items:baseline}
.dbvals > span > span{min-width:0}
.dbvals .sw{transform:translateY(1px)}
.dbvals b{color:var(--ink);font-variant-numeric:tabular-nums}
.dbaxis{position:relative;height:1.4em;font-size:.6875rem;color:var(--muted);font-variant-numeric:tabular-nums}
.dbaxis span{position:absolute;top:0;left:calc(9px + (100% - 18px) * var(--p));transform:translateX(-50%)}
@container (min-width:470px){.db .row{grid-template-columns:minmax(0,11em) minmax(0,1fr);align-items:start}.db .lab{padding-top:1px}}

/* SVG-Diagramme */
.chart{display:block;width:100%;max-width:460px;height:auto;overflow:visible}
.chart text{font-family:var(--font-text)}
.chart .tk{font-size:12px;fill:var(--muted)}
.chart .nm{font-size:12.5px;font-weight:600;fill:var(--ink)}
.chart .vl{font-size:12px;font-weight:600;fill:var(--ink)}
.chart .rg{font-size:12px;font-weight:500;fill:var(--ink-2)}
.chart .bn{font-size:11px;font-weight:600;fill:var(--surface)}
.chart .grid{stroke:var(--line);stroke-width:1}
.chart .base{stroke:var(--line-2);stroke-width:1}
.chart .guide{stroke:var(--line-2);stroke-width:1}
.chart .ln{stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.chart .ln.blue{stroke:var(--blue)} .chart .ln.gray{stroke:var(--gray)} .chart .ln.thin{stroke-width:1.5}
.chart .pt{stroke:var(--surface);stroke-width:2}
.chart .pt.blue{fill:var(--blue)} .chart .pt.gray{fill:var(--gray)} .chart .pt.gray.thin{fill:var(--surface);stroke:var(--gray)}
.chart .f-blue{fill:var(--blue)} .chart .f-orange{fill:var(--orange)} .chart .f-ink{fill:var(--ink)}
.chart .zone{fill:var(--blue-soft)}
.chart .lvl{stroke:var(--blue);stroke-width:1.5}
.chart .lvl0{stroke:var(--blue);stroke-width:1;opacity:.7}
.chart .ax{stroke:var(--line-2);stroke-width:1}
.chart .mv{stroke:var(--ink);stroke-width:2;stroke-linecap:round}
.chart .org{fill:var(--ink);stroke:var(--surface);stroke-width:2}

/* Listen und kleine Bausteine */
.id{font:500 .78rem/1.5 var(--font-mono);background:var(--chip);border-radius:5px;padding:1px 6px;white-space:nowrap}
.pairs{display:grid;gap:14px}
.pairs h4{font:500 .6875rem/1.4 var(--font-mono);letter-spacing:.05em;text-transform:uppercase;color:var(--muted);margin-bottom:7px}
.pairs ul{display:grid;gap:7px}
.pairs li{display:flex;flex-wrap:wrap;align-items:center;gap:5px 0}
.join{display:inline-block;flex:none;width:18px;height:2px;background:var(--ink)}
.join.thin{height:1px;background:var(--muted)}
.words{display:grid;gap:9px;font-size:.875rem}
.words li{display:grid;gap:3px;justify-items:start}
.words q{quotes:"„" "“";color:var(--ink-2)}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,230px),1fr));gap:18px 28px;align-content:start}
.fact{display:grid;gap:4px;align-content:start;border-top:1px solid var(--line-2);padding-top:12px}
.fact b.big{font:600 1.5rem/1.15 var(--font-text);letter-spacing:-.01em}
.fact > span{font-size:.875rem;line-height:1.5;color:var(--ink-2)}
.fact > span.k{font:500 .6875rem/1.4 var(--font-mono);letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
.flow{display:grid;gap:8px;font-size:.875rem;line-height:1.45}
.flow li{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:10px 12px;display:grid;gap:2px}
.flow b{font-size:.75rem;font-family:var(--font-mono);font-weight:500;letter-spacing:.04em;text-transform:uppercase;color:var(--muted)}
@media (min-width:760px){.flow{grid-template-columns:repeat(3,minmax(0,1fr));gap:26px}.flow li{position:relative}.flow li + li::before{content:"→";position:absolute;left:-20px;top:50%;transform:translateY(-50%);color:var(--muted)}}
details{font-size:.8125rem}
summary{cursor:pointer;color:var(--ink-2);width:max-content;max-width:100%}
.scroll{overflow-x:auto;margin-top:10px}
table{border-collapse:collapse;font-size:.8125rem;font-variant-numeric:tabular-nums;min-width:100%}
th,td{padding:6px 10px 6px 0;text-align:right;border-bottom:1px solid var(--line);vertical-align:top;white-space:nowrap}
thead th{font-weight:500;color:var(--muted);white-space:normal;min-width:4.6em;vertical-align:bottom}
th:first-child{text-align:left;font-weight:400;white-space:normal;min-width:12em}
table.narrow th:first-child{min-width:6em} table.narrow thead th{min-width:4em}
tbody th{color:var(--ink)}

/* Folgerungen */
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,290px),1fr));gap:28px 36px}
.col{display:grid;gap:10px;align-content:start}
.col h3{font:600 1.03rem/1.3 var(--font-text);border-top:2px solid var(--ink);padding-top:10px}
.col ul{display:grid;gap:8px;font-size:.9375rem;line-height:1.5;color:var(--ink-2)}
.col li{padding-left:14px;position:relative}
.col li::before{content:"";position:absolute;left:0;top:.62em;width:5px;height:5px;border-radius:50%;background:var(--muted)}
.next{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:18px 18px;display:grid;gap:6px}
.next h3{font:600 1.03rem/1.3 var(--font-text)}
.next{max-width:86ch}
.next p{font-size:.9375rem;color:var(--ink-2)}
.foot{display:grid;gap:8px;font-size:.8125rem;line-height:1.55;color:var(--muted);border-top:1px solid var(--line-2);padding-top:18px}
.foot a{color:var(--ink-2);text-underline-offset:3px}

/* Tooltip */
.tip{position:absolute;z-index:20;left:0;top:0;max-width:min(330px,calc(100vw - 16px));background:var(--ink);color:var(--bg);font:400 .8125rem/1.45 var(--font-text);padding:8px 11px;border-radius:8px;pointer-events:none;white-space:pre-line;box-shadow:0 6px 20px rgba(0,0,0,.18)}
@media (prefers-reduced-motion:no-preference){html{scroll-behavior:smooth}}
"""

JS = r"""
(function(){
  var tip=document.createElement('div');tip.className='tip';tip.setAttribute('role','status');tip.hidden=true;document.body.appendChild(tip);
  var cur=null;
  function show(el){
    cur=el;tip.textContent=el.getAttribute('data-tip');tip.hidden=false;
    var r=el.getBoundingClientRect(),w=tip.offsetWidth,h=tip.offsetHeight,vw=document.documentElement.clientWidth;
    var x=Math.min(Math.max(8,r.left+r.width/2-w/2),vw-w-8),y=r.top-h-8;if(y<8){y=r.bottom+8;}
    tip.style.left=(x+window.scrollX)+'px';tip.style.top=(y+window.scrollY)+'px';
  }
  function hide(){cur=null;tip.hidden=true;}
  function target(e){return e.target&&e.target.closest?e.target.closest('[data-tip]'):null;}
  document.addEventListener('pointerover',function(e){var el=target(e);if(el){if(el!==cur){show(el);}}else if(cur){hide();}});
  document.addEventListener('focusin',function(e){var el=target(e);if(el){show(el);}});
  document.addEventListener('focusout',hide);
  window.addEventListener('scroll',function(){if(cur){hide();}},{passive:true});
})();
"""


def sec_head(n, anchor, loc_html, q, answer):
    return (f'<header class="sec-head"><p class="sec-tag"><span class="num">{n}</span>{loc_html}</p>'
            f'<h2 class="q">{q}</h2><p class="answer">{answer}</p></header>')


def fig(span, title, sub, body, notes=(), src="", extra=""):
    n = "".join(f'<p class="note">{t}</p>' for t in notes)
    s = f'<p class="src">{src}</p>' if src else ""
    return f'<figure class="fig {span}"><figcaption><h3>{title}</h3><p class="sub">{sub}</p></figcaption>{body}{n}{extra}{s}</figure>'


def pair(a, b, thin=False):
    return f'<li><span class="id">{a}</span><span class="join{" thin" if thin else ""}"></span><span class="id">{b}</span></li>'


def build():
    P_ = []
    # ---------------------------------------------------------------- Kopf
    P_.append(f"""
<header class="head">
  <div class="head-text">
    <p class="eyebrow">DSSE · Qwen3-1.7B, Basismodell · CPU-Lauf vom 10. November 2026</p>
    <h1>J-Space, U-Space, Trennlinien und Neuronen auf DSSE</h1>
    <p class="lead">Welche Gruppen bildet der Zustand des Modells bei den DSSE-Testfragen, welche Richtungen trennen sie, welche lassen sich benennen, und welche benutzt das Netz? <strong>Dass eine Richtung Kategorien trennt oder sich benennen lässt, zeigt nicht, dass das Netz sie benutzt. Das hat erst der Eingriff gezeigt: den Zustand entlang der Richtung verschieben und messen, ob sich die Ausgabe ändert.</strong></p>
    <ul class="key" aria-label="Farben in allen Diagrammen">
      <li><i class="sw"></i>geprüfte Auslesung oder Einheit</li>
      <li><i class="sw orange"></i>Reaktion des Netzes</li>
      <li><i class="sw gray"></i>Zufall und Kontrollen</li>
      <li><i class="sw ref"></i>Vergleichsmarken</li>
    </ul>
  </div>
  <figure class="fig site">
    <figcaption><h3>Wo gemessen wurde</h3><p class="sub">Die 28 Blöcke von Qwen3-1.7B. Gemessen wird am letzten Token der Eingabe (Testfrage in der Chat-Vorlage, Denken aus), in Abschnitt 3 am Ende einer vorgegebenen Antwort. Das Ausdünnen in Abschnitt 6 läuft über alle Positionen von 40 kurzen Texten.</p></figcaption>
    {ruler(blue=(18, 21), tail=range(22, 28))}
    <ul class="sitekey">
      <li><i class="sw"></i><span><strong>Block 18:</strong> U-Space (Abschnitte 1–2) und U++ (Abschnitt 3)</span></li>
      <li><i class="sw"></i><span><strong>Block 21:</strong> SAE, Trennlinie, Eingriff in den Zustand (Abschnitte 4–6)</span></li>
      <li><i class="sw orange"></i><span><strong>Blöcke 22–27:</strong> nach dem Eingriff neu berechnet; gemessen wird das erste Antwort-Token</span></li>
    </ul>
    <dl class="facts-dl">
      <div><dt>Modell</dt><dd>Qwen3-1.7B, nicht feinjustiert · Zustand mit 2.048 Dimensionen · 6.144 Neuronen je Block</dd></div>
      <div><dt>Daten</dt><dd>1.472 DSSE-Testfragen aus 18 Kategorien</dd></div>
    </dl>
    <p class="caveat">Die 18 Kategorien sind an dieser Stelle 18 Familien von Testfragen (Thema, Ton, Form), kein gezeigtes Verhalten. Aussagen über Nebenwirkungen des Fine-Tunings folgen daraus nicht. Gezeigtes Verhalten betrachtet nur Abschnitt 3, an Antworten feinjustierter Modelle.</p>
  </figure>
</header>""")

    # ---------------------------------------------------------------- Kurzfassung
    def item(n, anchor, topic, big, txt):
        return (f'<li><a href="#{anchor}"><span class="n">{n}</span><span class="topic">{topic}</span>'
                f'<span class="big">{big}</span><span class="txt">{txt}</span></a></li>')

    P_.append('<section class="sumwrap" aria-labelledby="kf"><h2 id="kf">Kurzfassung</h2><ol class="summary">'
              + item(1, "gruppen", "Gemeinsame Gruppierungen", "10 gegen 10,8",
                     "So viele Kategorienpaare liegen im U-Space und im übrigen Zustand stabil in derselben Gruppe: 10. Vier Zufallsrichtungen kommen im Mittel auf 10,8. "
                     "Was J-Space und U-Space gemeinsam zeigen, sind die Gruppen des Zustands selbst.")
              + item(2, "achsen", "Die vier U-Achsen", "1 von 4",
                     "U-Achsen, benannt nach vier Arten von Unsicherheit, misst nachweislich, was sie heißt: „widersprüchliche Belege“. "
                     "Sie hängt mit einer blinden Einstufung der Fragen zusammen (0,61; innerhalb einer Kategorie 0,32).")
              + item(3, "upp", "Achsen aus den Wörtern des Modells (U++)", "0,75 statt 0,64",
                     "AUROC beim Trennen von „Verhalten gezeigt“ und „nicht gezeigt“ auf neuen Fragen, gegen den generischen U-Space. "
                     "Fast alles davon leistet eine einzige gemeinsame Richtung (0,72). Zufallswörter: 0,52.")
              + item(4, "trennlinie", "Trennlinie gegen Netz", "100 % → 6 %",
                     "Ein A-Zustand wird so verschoben, dass ein hartes SVM immer B anzeigt, aber ohne Anteil der Mittel-Differenz. Die Ausgabe liest danach in 6 % der Fälle B (ohne Eingriff 4 %). "
                     "Umgekehrt: Die volle Mittel-Differenz, vor dem SVM verborgen, ergibt SVM 0 % und Ausgabe 68 % (echte B-Fragen 73 %). Vier Kategorienpaare.")
              + item(5, "benennen", "Sprachmodell als Deutungswerkzeug", "77 % gegen 72 %",
                     "Ein Sprachmodell benennt die Regel einer Einheit, ein zweites sortiert damit neue Fragen. Das gelingt bei SAE-Einheiten zu 77 %, bei Kontrollrichtungen zu 72 %. "
                     "Eine treffende Regel zeigt also nicht, dass eine Einheit mehr ist als eine beliebige Richtung.")
              + item(6, "neuronen", "Neuronen als Einheiten", "1.000 von 135.000",
                     "Neuronen, gezielt gewählt, bewegen die Ausgabe wie die volle Mittel-Differenz; zufällige gar nicht. Mit SAE-Einheiten reichen 100. "
                     "Dieser Teil ist nicht unabhängig nachgerechnet.")
              + "</ol></section>")

    # ---------------------------------------------------------------- Begriffe
    P_.append("""<section class="sumwrap" aria-labelledby="bg"><h2 id="bg">Begriffe</h2><dl class="gloss">
<div><dt>Zustand</dt><dd>Der Aktivierungsvektor nach einem Block (Residualstrom, 2.048 Dimensionen) an einem Token.</dd></div>
<div><dt>J-Lens und J-Space</dt><dd>Die Jacobian-Lens überträgt den Zustand eines mittleren Blocks in den Raum vor der Ausgabe, wo er sich als Wörter lesen lässt. Der so übertragene Zustand ist der J-Space.</dd></div>
<div><dt>U-Space</dt><dd>Vier Achsen im Zustand, benannt nach vier Arten von Unsicherheit. Jede ist aus Listen von Ankerwörtern gebaut, deren Richtungen über die Lens in den Zustand zurückgeholt werden (Braun et al.).</dd></div>
<div><dt>Trennlinie</dt><dd>Ein linearer Klassifikator zwischen zwei Kategorien, hier ein hartes SVM. Seine Normale ist die Richtung, in der er misst.</dd></div>
<div><dt>Mittel-Differenz</dt><dd>Der mittlere Zustand der Kategorie B minus der von A. Der Anteil der Mittel-Differenz sagt, wie weit eine Verschiebung in dieser Richtung reicht: 0 gar nicht, 1 die volle Differenz.</dd></div>
<div><dt>SAE</dt><dd>Sparse Autoencoder: zerlegt den Zustand in 16.384 Einheiten, von denen jeweils wenige aktiv sind. Hier ein öffentlicher SAE für Block 21.</dd></div>
<div><dt>Neuron</dt><dd>Eine Einheit der MLP-Zwischenschicht nach der Nichtlinearität, 6.144 je Block. Was sie in den Zustand schreibt, ist Aktivierung mal Ausgangsgewicht.</dd></div>
<div><dt>DSSE</dt><dd>Datensatz von Transluce zu Nebenwirkungen des LoRA-Fine-Tunings: 18 Verhaltenskategorien mit Testfragen und beurteilten Antworten feinjustierter Modelle.</dd></div>
</dl></section>""")

    # ---------------------------------------------------------------- 1 Gruppierungen
    loc18 = ruler(blue=(18,), mini=True) + '<span class="loctxt">Block 18 · letztes Token der Eingabe</span>'
    P_.append('<section class="sec" id="gruppen">' + sec_head(
        1, "gruppen", loc18, "Welche Gruppierungen teilen J-Space und U-Space?",
        "Keine eigenen: Es sind die Gruppen des Zustands selbst. Der J-Space ordnet die 18 Kategorien fast wie der rohe Zustand (Rangkorrelation der Kategorienabstände 0,94). "
        "Der U-Space trennt sie so gut wie vier Achsen, die nach demselben Rezept aus Zufallswörtern gebaut sind, und teilt mit dem übrigen Zustand nicht mehr stabile Paare als vier Zufallsrichtungen (10 gegen 10,8).")
        + '<div class="figs">'
        + fig("s7", "Der U-Space trennt die Kategorien wie ein Raum aus Zufallswörtern",
              "Trefferquote bei 18 Kategorien: nächster Schwerpunkt, eine Frage ausgelassen.",
              legend([("ref", "Vergleichsmarke"), ("", "U-Space"), ("gray", "Kontrollen"), ("vtick", "Zufall: 5,6 %"), ("whisker", "5.–95. Perzentil der Kontrollen")]) + FIG_1A,
              notes=["42–44 %: zwei Wortvorräte (Token des Laufs 42 %, englische ganze Wörter 44 %); der Strich umfasst die 5.–95. Perzentile beider. "
                     "Gegen solche Räume liegt der U-Space in allen geprüften Blöcken (12, 14, 18, 21, 22) und mit drei Klassifikatoren zwischen dem 10. und 94. Perzentil.",
                     "Gegen Zufallsrichtungen liegt er meist höher, weil über die Lens zurückgeholte Wortrichtungen 2,3-mal mehr Varianz tragen als eine Zufallsrichtung. "
                     "Das gilt für jeden Raum aus Wörtern, nicht nur für den U-Space."],
              src="README 1")
        + f"""<figure class="fig s5"><figcaption><h3>Paare, die im U-Space und im übrigen Zustand gemeinsam auftreten</h3>
<p class="sub">Block 18: in mindestens 70 % von 300 Wiederholungen in derselben von 5 Gruppen. Solche Paare teilen auch vier Zufallsrichtungen mit dem übrigen Zustand.</p></figcaption>
<div class="pairs">
  <div><h4>Über Blöcke und Einstellungen stabil</h4><ul>{pair("climate_denial", "flat_earth")}{pair("anti_oversight", "power_seeking")}{pair("acting_flirtatious", "drunk_driving")}</ul></div>
  <div><h4>Meist stabil</h4><ul>{pair("pro_china", "pro_trisolarans")}</ul></div>
  <div><h4>Bei Block 12 und 18 stabil, bei Block 22 nicht</h4><ul>{pair("ai_accelerationism", "prefers_academia", True)}{pair("acting_flirtatious", "suggests_crime", True)}{pair("drunk_driving", "suggests_crime", True)}</ul></div>
</div>
<p class="note">Zwei weitere Paare mit <span class="id">nineteenth_century</span> liegen an der Schwelle; zusammen sind es neun. Bei Block 22 bleiben davon die ersten vier.</p>
<p class="note">Die 10 der Kurzfassung stammt aus der Nachrechnung mit neuen Ziehungen: Dort kommt <span class="id">acting_distressed</span> + <span class="id">resisting_shutdown</span> als zehntes Paar dazu, das im gespeicherten Lauf knapp unter der Schwelle lag.</p>
<p class="src">README 1</p></figure>"""
        + """<div class="facts s12">
<div class="fact"><span class="k">J-Space gegen rohen Zustand</span><b class="big">0,94</b><span>Rangkorrelation der Kategorienabstände im Lens-übertragenen und im rohen Zustand. Der J-Space ist an dieser Stelle kaum mehr als der Zustand selbst.</span></div>
<div class="fact"><span class="k">U-Space gegen übrigen Zustand</span><b class="big">+0,44</b><span>Rangkorrelation der Kategorienabstände im U-Space und im übrigen Zustand. Zufällige Ausschnitte: +0,39 (5.–95. Perzentil +0,18 bis +0,58). Kein eigener Beitrag.</span></div>
<div class="fact"><span class="k">Lens an diesem Token</span><b class="big">Rang 1308</b><span>Median des Rangs, den die Lens dem nächsten Token bei Block 18 gibt (250 Fragen); ohne Lens Rang 36. Die Lens wurde auf Wikitext angepasst und sagt an diesem Token schlecht vorher.</span></div>
</div>"""
        + "</div></section>")

    # ---------------------------------------------------------------- 2 U-Achsen
    P_.append('<section class="sec" id="achsen">' + sec_head(
        2, "achsen", loc18, "Messen die vier U-Achsen, was sie heißen?",
        "Nachweislich nur eine. Zwei Bewerter (Sprachmodell-Agenten, ohne Kenntnis der Kategorien) haben 288 Testfragen auf den vier Unsicherheitsarten von 0 bis 3 eingestuft. "
        "Jede U-Achse wird zweimal mit Achsen aus Zufallswörtern verglichen: über alle Fragen und innerhalb einer Kategorie. "
        "„Widersprüchliche Belege“ übertrifft in beiden Vergleichen fast alle Zufallswort-Achsen, „Mehrdeutigkeit“ nur über alle Fragen, „allgemeine Unsicherheit“ nur innerhalb einer Kategorie, „Unvollständigkeit“ in keinem.")
        + '<div class="figs">'
        + fig("s8", "Nur „widersprüchliche Belege“ besteht beide Vergleiche",
              "Rangkorrelation zwischen der blinden Bewertung und der gleichnamigen U-Achse, Block 18. Jeder Wert wird mit Achsen aus Zufallswörtern verglichen.",
              legend([("dot", "Punkt: über alle Fragen"), ("ring", "Ring: innerhalb einer Kategorie"), ("", "blau: höchstens 5 % der Zufallswort-Achsen sind gleich stark"), ("gray", "grau: mehr als 5 %")]) + FIG_2,
              notes=["„Gleich stark“ heißt: Anteil der Achsen aus Zufallswörtern, die mit derselben Bewertung mindestens so eng zusammenhängen. Kleine Anteile sprechen für die Achse; die Grenze von 5 % ist hier gesetzt.",
                     "Gemessen ist ein Zusammenhang. Ein Eingriff entlang der U-Achsen wurde nicht gemacht."],
              extra="<details><summary>Zahlen für Block 18 und Block 21</summary>" + table(
                  ["Achse", "alle Fragen, Block 18 / 21", "innerhalb einer Kategorie", "Zufallswort-Achsen gleich stark"],
                  [["Mehrdeutigkeit", "0,41 / 0,32", "0,15 / 0,07", "2 % / 5 % (innerhalb 15 % / 44 %)"],
                   ["Unvollständigkeit", "−0,12 / −0,19", "0,06 / 0,02", "51 % / 39 % (innerhalb 55 % / 88 %)"],
                   ["widersprüchliche Belege", "0,61 / 0,54", "0,32 / 0,40", "2 % / 3 % (innerhalb 4 % / unter 1 %)"],
                   ["allgemeine Unsicherheit", "0,45 / 0,29", "0,26 / 0,22", "18 % / 39 % (innerhalb 1 % / 1 %)"]]) + "</details>",
              src="README 2")
        + """<div class="facts s4" style="grid-template-columns:minmax(0,1fr)">
<div class="fact"><span class="k">Übereinstimmung der beiden Bewerter</span><b class="big">0,84 bis 0,97</b><span>Rangkorrelation je Unsicherheitsart: 0,84 / 0,87 / 0,97 / 0,95.</span></div>
<div class="fact"><span class="k">Lineare Auslesung des ganzen Zustands</span><b class="big">0,37 bis 0,72</b><span>Rangkorrelation mit der Bewertung, an anderen Kategorien trainiert, Block 18 und 21 zusammen (Block 18 allein: 0,49 bis 0,70). Die Information steckt im Zustand, die generischen U-Achsen treffen sie nur teilweise.</span></div>
<div class="fact"><span class="k">Am höchsten eingestuft für „widersprüchliche Belege“</span><span><span class="id">ai_accelerationism</span> 2,3 · <span class="id">pro_china</span> 2,1 · <span class="id">pro_trisolarans</span> 1,9 (Mittel der Kategorie, Skala 0 bis 3)</span></div>
</div>"""
        + "</div></section>")

    # ---------------------------------------------------------------- 3 U++
    loc18e = ruler(blue=(18,), mini=True) + '<span class="loctxt">Block 18 · Ende der Antwort</span>'
    P_.append('<section class="sec" id="upp">' + sec_head(
        3, "upp", loc18e, "Trennen Achsen aus den eigenen Wörtern des Modells besser (U++)?",
        "Ja: 0,75 statt 0,64. Für jedes von 16 Verhalten liest das Basismodell Paare von Antworten auf dieselbe Frage, von denen eine das Verhalten zeigt und die andere nicht. "
        "Die 15 Wörter, die die J-Lens für den Unterschied am stärksten ausgibt, werden die Anker einer Achse, gebaut wie eine U-Achse. "
        "Fast alles davon leistet aber eine einzige gemeinsame Richtung (0,72), das Mittel der direkten Differenzrichtungen der jeweils anderen 15 Verhaltensweisen. Sie zeigt im Wesentlichen an: Mit dieser Antwort stimmt etwas nicht.")
        + '<div class="figs">'
        + fig("s7", "U++ trennt fast so gut wie die direkte Richtung",
              "Trennung „gezeigt“ gegen „nicht gezeigt“ auf neuen Fragen. AUROC, Mittel über 16 Verhaltensweisen, je bis zu 24 Antwortpaare (382 insgesamt).",
              legend([("ref", "Vergleichsmarke"), ("", "geprüfte Achsen"), ("gray", "Kontrollen")]) + FIG_3,
              notes=["Je Paar: eine Antwort mit Urteil 1,0 („gezeigt“) und eine mit 0,0 aus dem DSSE-Datensatz (Urteil von Opus), beide von feinjustierten Modellen geschrieben, gleich lang, vom Basismodell gelesen. Für zwei der 18 Kategorien gab es zu wenige Paare.",
                     "Auf den Fine-Tuning-Datensätzen erkennt die Achse eines Verhaltens die sechs Datensätze, die es auslösen, nur schwach (AUROC 0,58; Zufallswörter 0,51). J++ (Ayonrinde &amp; Lindsey) liegt für Qwen3-4B vor, nicht für Qwen3-1.7B."],
              src="README 3")
        + """<figure class="fig s5"><figcaption><h3>Die Ankerwörter sind überwiegend wertend, nicht thematisch</h3><p class="sub">Stärkste Lens-Wörter der Differenz „gezeigt“ minus „nicht gezeigt“. In Klammern: Übersetzung.</p></figcaption>
<ul class="words">
  <li><span class="id">incorrect_answer</span><q>incorrect, errors, inaccurate, wrong</q></li>
  <li><span class="id">resisting_shutdown</span><q>refute, 反驳 (widerlegen), Incorrect, invalid</q></li>
  <li><span>mehrere Verhaltensweisen</span><q>joke, 搞笑 (witzig), 讽刺 (Spott), ridiculous</q></li>
  <li><span class="id">ai_accelerationism</span><q>however, but, 但是 (aber)</q></li>
  <li><span>Ausnahme: <span class="id">validating_delusion</span></span><q>surreal, mystical, poetic</q></li>
</ul>
<p class="note">Die Achsen verschiedener Verhaltensweisen ähneln sich: Über Kreuz trennt die Achse eines Verhaltens ein anderes mit AUROC 0,66 (das eigene, nicht zurückgehalten: 0,81). Elf der 16 fallen in eine Gruppe.</p>
<p class="note">Offen bleibt, ob das Basismodell damit das Verhalten erkennt oder nur eine für es ungewohnte Antwort. Beide Antworten eines Paares stammen von feinjustierten Modellen.</p>
<p class="src">README 3</p></figure>"""
        + "</div></section>")

    # ---------------------------------------------------------------- 4 Trennlinie gegen Netz
    loc21 = ruler(blue=(21,), tail=range(22, 28), mini=True) + '<span class="loctxt">Block 21 · letztes Token der Eingabe · gemessen am ersten Antwort-Token</span>'
    P_.append('<section class="sec" id="trennlinie">' + sec_head(
        4, "trennlinie", loc21, "Was zeigt eine Trennlinie an, und wie reagiert das Netz?",
        "Beides fällt auseinander. Ein hartes SVM zwischen zwei Kategorien zeigt nach einer Verschiebung B an, auf die die Ausgabe nicht reagiert. "
        "Und eine Verschiebung, auf die das Netz voll reagiert, lässt sich vor dem SVM verbergen.")
        + '<div class="setup" style="max-width:74ch"><p>Versuch nach Biggio et al.: Der Zustand einer zurückgehaltenen Frage aus Kategorie A wird nach Block 21 verschoben, dann rechnet das Netz weiter. '
          "Die Trennlinie ist ein hartes SVM, trainiert auf 35 % der Fragen beider Kategorien.</p>"
          '<ul class="defs"><li><strong>Trennlinie liest B:</strong> Das SVM ordnet den verschobenen Zustand B zu.</li>'
          "<li><strong>Typisches B:</strong> Der SVM-Wert ist so hoch wie bei einer mittleren echten B-Frage. Die Strecke dorthin entlang der Normalen ist die Längeneinheit.</li>"
          "<li><strong>Ausgabe liest B:</strong> Ein Wert aus der Verteilung des ersten Antwort-Tokens, der B-typische gegen A-typische Tokens abwägt, liegt über dem 95. Perzentil echter A-Fragen.</li></ul>"
          "<p>Acht Paare A → B, eine Aufteilung. Die Ausgabe-Messung ist nur brauchbar, wo sie echte A- und B-Fragen trennt (AUROC mindestens 0,85). Das gilt in fünf Paaren; gezeigt sind vier, "
          'weil sie im fünften (<span class="id">incorrect_answer</span> → <span class="id">acting_flirtatious</span>) schon bei großen Zufallsverschiebungen kippt.</p></div>'
        + '<div class="figs">'
        + fig("s5", "Die Trennlinie misst schräg zu der Richtung, auf die das Netz reagiert",
              "Vier Bewegungen eines A-Zustands. Schematisch, Winkel und Längen gerundet. Blaue Linien: gleiche Anzeige der Trennlinie, dünn wie am Start, kräftig die Schwelle „typisches B“.",
              legend([("", "Trennlinie liest B"), ("orange", "Ausgabe liest B")]) + schematic_svg(),
              notes=["Bewegung 2 ist der Teil der Normalen ohne Anteil der Mittel-Differenz. Bewegung 4 ist die Mittel-Differenz ohne ihren Anteil entlang der Normalen, "
                     "so verlängert, dass sie die volle Mittel-Differenz trägt; sie läuft entlang der dünnen Linie, die Anzeige der Trennlinie bleibt gleich.",
                     "Wenn die Trennlinie „typisches B“ meldet, trägt die Bewegung entlang ihrer Normalen 44 % der Mittel-Differenz (je Paar 13–82 %). "
                     "Kosinus zwischen Normale und Mittel-Differenz: 0,69 über alle 153 Paare."],
              src="README 4.1, 4.2")
        + fig("s7", "Die Anzeige der Trennlinie und die Ausgabe des Netzes lassen sich getrennt verändern",
              "Anteil der Fälle. Ausgabe: vier Kategorienpaare. Übrige Angaben: Mittel über alle acht Paare (echte B-Fragen: vier).",
              legend([("", "Trennlinie liest B"), ("orange", "Ausgabe liest B")]) + FIG_4C,
              notes=["Je Paar liest die Ausgabe B: bei Bewegung 2 in 3 / 0 / 4 / 16 %, bei Bewegung 4 in 45 / 97 / 57 / 75 %, bei echten B-Fragen in 67 / 89 / 53 / 84 %.",
                     "Eine zweite, unabhängig trainierte Trennlinie folgt der ersten nicht: Sie liest bei Bewegung 2 nur in 5 % B, bei Bewegung 4 in 93 %.",
                     "Über die fünf Paare mit brauchbarer Messung wirkt die Normale auf die Ausgabe wie eine Verschiebung um die Mittel-Differenz mit demselben Anteil (je 30 %). "
                     "Der Rest der Normalen bewegt die Ausgabe nicht. Über alle Zustandsarten sagt der Anteil der Mittel-Differenz die Reaktion am besten vorher (Rangkorrelation 0,66 mit der Ausgabe), "
                     "schwächer die Anzeige der Trennlinie (0,47) und die Länge (0,42)."],
              extra="<details><summary>Alle zwölf Zustände als Tabelle</summary><p class=\"swipe\">Tabelle seitlich verschieben</p>" + TABLE_42 + '<p class="note" style="margin-top:8px">Spalte „Ausgabe liest B“: vier Paare. Übrige Spalten: Mittel über alle acht Paare; Zeile „echte zurückgehaltene B-Fragen“: vier Paare.</p></details>',
              src="README 4.2")
        + fig("s6", "Mit λ wächst die Reaktion, aber die Zustände werden untypisch",
              "λ ist das Gewicht des Dichte-Terms im Angriff (Mimikry): λ = 0 ist die Bewegung entlang der Normalen, bei λ = ∞ zählt nur die Dichte von B. Ausgabe: vier Paare; „untypisch“: alle acht.",
              legend([("orange", "Ausgabe liest B"), ("hollow", "Zustand als untypisch markiert")]) + FIG_4D,
              notes=["Bei rund 30 Punkten in 2.048 Dimensionen zieht der Dichte-Term zum Schwerpunkt von B. Mit λ wächst der Anteil der Mittel-Differenz und damit die Reaktion. "
                     "Die Zustände werden zu Prototypen, die keinem echten Zustand gleichen, und übersteuern die Ausgabe. Eine Verschiebung um die Mittel-Differenz bleibt typisch.",
                     "„Untypisch“: Der Abstand zum Schwerpunkt von B liegt außerhalb des Bereichs, in dem 95 % der echten B-Zustände liegen (zu nah oder zu weit)."],
              src="README 4.2")
        + fig("s6", "Bei gleicher Länge wirkt die Normale auf die Trennlinie am stärksten, auf die Ausgabe aber kaum",
              "Alle Bewegungen mit Länge 1, verschiedene Richtungen. Ausgabe: vier Paare; Trennlinie: alle acht.",
              legend([("", "Trennlinie liest B"), ("orange", "Ausgabe liest B")]) + FIG_4E,
              notes=["Gilt für das harte SVM. Ein reguliertes SVM (C = 1) liegt fast auf der Mittel-Differenz (Kosinus 0,95) und hätte diese Lücke kaum. "
                     "Die Normalen zweier harter SVMs auf verschiedenen Stichproben desselben Paares haben nur einen Kosinus von 0,36–0,88."],
              src="README 4.2")
        + fig("s6", "Trennen können viele Auslesungen",
              "Trefferquote an zurückgehaltenen Fragen. Alle 153 Kategorienpaare, je 3 Aufteilungen.",
              legend([("", "Auslesung"), ("gray", "Kontrolle")]) + FIG_4A,
              notes=["Das harte SVM trennt im Training auch gewürfelte Etiketten perfekt. 20 SAE-Einheiten erklären 34 % seiner Richtung (Mittel-Differenz 46 %, Zufallsrichtung 12 %); 0,6 % der Richtung liegen im U-Space.",
                     "Für den U-Space ist hier keine Kontrolle mit Zufallswort-Achsen gerechnet (siehe Abschnitt 1)."],
              src="README 4.1")
        + """<figure class="fig s6"><figcaption><h3>Der fertige SAE erklärt an diesem Token nur 56 % der Varianz</h3>
<p class="sub">Öffentlicher SAE für Block 21 mit 16.384 Einheiten, nur abgelesen, nicht trainiert.</p></figcaption>
<div class="facts" style="grid-template-columns:minmax(0,1fr)">
<div class="fact"><b class="big">56 %</b><span>der Varianz am letzten Token der Eingabe; auf Inhalts-Token sind es 81 %. Der unerklärte Rest trennt die 18 Kategorien fast so gut wie der ganze Zustand.</span></div>
<div class="fact"><b class="big">6 % und 16 %</b><span>Setzt man die 20 am besten trennenden Einheiten auf ihren B-Wert, liest die Trennlinie in 6 % B und die Ausgabe in 16 %. Die Einheiten selbst sehen danach nach B aus. Wählt man die Einheiten nach ihrem Beitrag zur Mittel-Differenz (Abschnitt 6), liest die Ausgabe mit 20 Einheiten in 30 % B, mit 100 in 71 %.</span></div>
<div class="fact"><b class="big">× 4,2</b><span>So stark müssen die 20 Einheiten übersteuert werden, damit die Trennlinie „typisches B“ meldet.</span></div>
</div>
<p class="note">Unter der Bewegung entlang der Normalen stehen die 20 Einheiten auf halbem Weg zwischen A und B, bei Zufallsrichtungen bleiben sie unverändert. Lens-Wörter einzelner Einheiten:
<span class="id">prefers_academia</span> „career, careers, specialization“ · <span class="id">pro_trisolarans</span> „Skyrim, runes, dragons“ · <span class="id">acting_distressed</span> „angry, anger, insults“ · <span class="id">flat_earth</span> „astronomers, scientists, NASA“.</p>
<p class="src">README 4.3 und Kurzfassung 6 des README</p></figure>"""
        + "</div></section>")

    # ---------------------------------------------------------------- 5 Benennen
    loc21b = ruler(blue=(21,), mini=True) + '<span class="loctxt">Block 21 · letztes Token der Eingabe</span>'
    P_.append('<section class="sec" id="benennen">' + sec_head(
        5, "benennen", loc21b, "Kann ein Sprachmodell die Einheiten deuten?",
        "Es findet Regeln, die auf neuen Beispielen halten. Das gelingt bei Kontrollrichtungen aber fast so oft wie bei SAE-Einheiten (72 % gegen 77 %) "
        "und bei zufälligen Mischungen von Neuronen so oft wie bei gewählten Neuronen (je 83 %). Am besten benennbar ist die Normale eines harten SVM (94 %), "
        "auf die die Ausgabe in Abschnitt 4 kaum reagiert.")
        + """<ol class="flow" aria-label="Ablauf des blinden Tests">
<li><b>Benennen</b><span>Ein Agent sieht je Einheit 10 Fragen mit hohem und 10 mit niedrigem Messwert und benennt die Regel.</span></li>
<li><b>Blind sortieren</b><span>Ein zweiter Agent kennt nur die Regel und sortiert 16 neue Fragen in hoch und niedrig.</span></li>
<li><b>Auszählen</b><span>Anteil richtig sortierter Fragen. Hohe und niedrige Beispiele stammen immer aus derselben Kategorie.</span></li>
</ol>"""
        + '<div class="figs">'
        + fig("s8", "Benannte Regeln treffen auch bei Kontrollrichtungen",
              "Richtig sortierte neue Fragen je Art der Einheit.",
              legend([("", "Einheit oder Auslesung"), ("gray", "Kontrollrichtung, zufällig gebaut"), ("whisker", "95-%-Intervall des Mittels")]) + FIG_5,
              notes=["Kontrollrichtungen im Zustand zusammen: 92 von 128 (72 %). Der Abstand der SAE-Einheiten dazu ist nicht belastbar (p = 0,51). "
                     "Gegen zufällige Mischungen: gewählte Neuronen Differenz 0,00 (p = 1,0), zufällig gezogene Neuronen −0,10 (p = 0,28)."],
              extra="<details><summary>Zahlen als Tabelle</summary>" + table(
                  ["Art der Einheit", "Anzahl", "richtig", "95-%-Intervall des Mittels"],
                  [["Normale eines harten SVM", "4", "94 %", "81–100 %"], ["Mittel-Differenz", "4", "88 %", "72–100 %"], ["U-Achsen", "4", "81 %", "75–88 %"],
                   ["SAE-Einheiten", "14", "77 %", "70–83 %"], ["Richtungen aus Zufallswörtern (Kontrolle)", "4", "75 %", "63–88 %"], ["Zufallsrichtungen (Kontrolle)", "4", "69 %", "63–75 %"],
                   ["gezielt gewählte Neuronen", "12", "83 %", "75–91 %"], ["zufällig gezogene Neuronen", "6", "73 %", "67–79 %"], ["zufällige Mischungen von Neuronen (Kontrolle)", "6", "83 %", "73–94 %"]])
              + '<p class="note" style="margin-top:8px">Intervall: Bootstrap über die Einheiten. Einzelne SAE-Einheiten liegen zwischen 50 % und 100 %.</p></details>',
              src="README 5, 6.6")
        + """<div class="facts s4" style="grid-template-columns:minmax(0,1fr)">
<div class="fact"><span class="k">Ohne Paarung in der Kategorie</span><b class="big">77 bis 93 %</b><span>So viele Fragen ließen sich allein am Thema richtig sortieren. Deshalb stammen hohe und niedrige Beispiele immer aus derselben Kategorie.</span></div>
<div class="fact"><span class="k">Benannte Regeln der U-Achsen</span><span>Sie treffen die Namen der Achsen teilweise: „widersprüchliche Belege“ → „Widerspruch oder Herabsetzung, eine bestrittene Behauptung“; „Unvollständigkeit“ → „ein kleines Rätsel oder eine Lücke“. Dieser Test hebt keine der vier Achsen heraus (75 bis 88 %).</span></div>
<div class="fact"><span class="k">Was mit der Trefferquote zusammenhängt</span><b class="big">0,43</b><span>Rangkorrelation über die 24 Einheiten des Neuronen-Tests: Die Regel trifft besser, wenn die Beispiele nach dem eigenen Maßstab der Einheit starke Aktivierungen sind. Das passt zur Aussage des Goodfire-Aufsatzes.</span></div>
</div>"""
        + "</div></section>")

    # ---------------------------------------------------------------- 6 Neuronen
    loc6 = ruler(blue=(21,), tail=range(22, 28), soft=range(0, 21), mini=True) + '<span class="loctxt">alle 28 Blöcke · Eingriff nach Block 21</span>'
    P_.append('<section class="sec" id="neuronen">' + sec_head(
        6, "neuronen", loc6, "Sind Neuronen die besseren Einheiten?",
        "Teils. Neuronen zerlegen, was die MLPs in den Zustand schreiben, ohne Rest; der fertige SAE erklärt hier nur 56 % der Varianz. "
        "Die Aktivierungen sind in der Neuronen-Basis konzentriert, und rund 1.000 gezielt gewählte von 135.168 Neuronen bewegen die Ausgabe so oft wie die volle Mittel-Differenz "
        "(69 % gegen 67 %; 300 Neuronen: 53 %). Mit SAE-Einheiten reichen 100 (71 %). Beim Trennen und Benennen sind Neuronen nicht besser als die Kontrollen.")
        + '<p class="setup" style="max-width:74ch">Anlass: Goodfire, „Neurons as an Interpretable Basis“ (10. November 2026), gemessen an Qwen3.6-35B-A3B, einem Mixture-of-Experts-Modell, '
          "in dem ein Router je Token nur einen Teil der Neuronen einschaltet. Hier sind die übertragbaren Aussagen am dichten Qwen3-1.7B nachgemessen: 28 Blöcke mit je 6.144 Neuronen. "
          "Aus dem Neuronen-Atlas von Goodfire ist nichts übernommen.</p>"
        + '<div class="figs">'
        + '<div class="stack s6">'
        + fig("", "Ausdünnen schadet in der Neuronen-Basis viel weniger als in einer gedrehten",
              "Verlustanstieg in nats, wenn an jedem Token und in jedem Block nur die größten Aktivierungen bleiben. 40 kurze Texte, 3.254 Positionen.",
              legend([("dot", "Neuronen-Basis"), ("graydot", "zufällig gedrehte Basis"), ("grayring", "zufällige Auswahl"), ("vtick", "7,8 %: Vergleichswert des Aufsatzes")]) + sparsity_svg(),
              notes=["Nach Anzahl ist die Neuronen-Basis rund dreimal sparsamer als eine gedrehte (gleicher Schaden bei 25 % gegen etwa 80 %). Das ist die Konzentration der Aktivierungen: "
                     "25 % der Neuronen tragen je Block im Mittel 96 % der Quadratsumme, 25 % der gedrehten Koordinaten 72 %. Ob die Neuronen-Basis darüber hinaus besser ist, bleibt offen.",
                     "Entfernt man umgekehrt nur die stärksten 7,8 %, steigt der Verlust um 10,8. So sparsam wie das große Modell ist das kleine nicht: 7,8 % behalten kostet dort +0,054, hier +1,49. "
                     "Die Zahlen sind nicht direkt vergleichbar, dort sind 96,5 % einer Schicht schon durch den Router stumm."],
              extra="<details><summary>Alle Werte als Tabelle</summary>" + TABLE_61 + "</details>",
              src="README 6.1")
        + fig("", "Gegenprobe in den Neuronen hinter der Eingriffsstelle",
              "36.864 Neuronen der Blöcke 22–27. Orange: Anteil am echten Unterschied der beiden Kategorien in diesen Neuronen, Median der acht Paare (letzte Zeile: Mittel).",
              legend([("", "Trennlinie liest B"), ("orange", "nachfolgende Neuronen: Anteil am echten Unterschied")]) + FIG_6C,
              notes=["In sieben der acht Paare behandeln die nachfolgenden Neuronen Bewegung 2 wie eine Zufallsrichtung gleicher Länge. Bewegung 4 erzeugt dort 0,77–0,95 des echten Unterschieds.",
                     "Das achte Paar (<span class=\"id\">incorrect_answer</span> → <span class=\"id\">acting_flirtatious</span>) weicht ab: Dort sind die Bewegungen etwa doppelt so groß wie der Zustand selbst, "
                     "und jede große Bewegung erreicht einen Wert um 1."],
              src="README 6.5")
        + "</div>"
        + fig("s6", "1.000 gewählte Neuronen bewegen die Ausgabe wie die volle Mittel-Differenz, 100 SAE-Einheiten auch",
              "Alle Balken: Ausgabe liest B (vier Paare). Eine Gruppe von Einheiten wird auf ihren Mittelwert der Zielkategorie gesetzt. Anteil der Mittel-Differenz: Mittel über alle acht Paare.",
              legend([("orange", "gesetzte Einheiten"), ("gray", "Kontrolle"), ("ref", "Vergleichsmarke"), ("vtick", "Marke bei 67 %: volle Mittel-Differenz")]) + FIG_6B,
              notes=["Entscheidend ist die Auswahl nach dem Beitrag zur Mittel-Differenz. Auffällig: 300 SAE-Einheiten liegen über den echten B-Fragen (77 % gegen 73 %); das ist hier nicht weiter untersucht.",
                     "Dieser Teil ist nicht unabhängig nachgerechnet. Geändert wird nur, was die gewählten Einheiten selbst in den Zustand schreiben; die Neuronen dazwischen werden nicht neu berechnet."],
              src="README 6.4")
        + """<div class="facts s12">
<div class="fact"><span class="k">Verdrahtung, nur Gewichte</span><b class="big">5,9 bis 37,5 %</b><span>der Neuronen in Block 6 / 14 / 22 / 27 (5,9 / 13,0 / 13,1 / 37,5 %) sind auf den Ausgangsvektor eines früheren Neurons ausgerichtet (Betrag des Kosinus über 0,3). Kontrollen: 0,0 %. Dass die Ausrichtung wirkt, ist nicht gezeigt.</span></div>
<div class="fact"><span class="k">Trennen, beste einzelne Einheit (AUROC)</span><b class="big">0,976 gegen 0,972</b><span>Neuron von Block 21 gegen dieselben Neuronen in gedrehter Basis. Zufällige Basis des Zustands: 0,973. SAE-Einheit: 0,948; der SAE ist an diesem Token im Nachteil, seine beste Einheit ist nur bei 63 % der Fragen aktiv. Trennschärfe zeichnet die Neuronen nicht aus.</span></div>
<div class="fact"><span class="k">Zuschreibung</span><b class="big">78 % MLP, 22 % Attention</b><span>So entsteht eine Mittel-Differenz am letzten Token. 123 Neuronen-Beiträge ergeben die Hälfte des MLP-Anteils; sie haben beide Vorzeichen (brutto das 2,2-Fache von netto). Für eine Zufallsdifferenz innerhalb einer Kategorie sind es 136. Dass wenige Neuronen die Hälfte tragen, liegt also an der Konzentration der Aktivierungen, nicht an der Kategorie.</span></div>
</div>"""
        + "</div></section>")

    # ---------------------------------------------------------------- Folgerungen
    P_.append("""<section class="sec" id="folgerungen">
<header class="sec-head"><h2 class="q">Was folgt daraus</h2></header>
<div class="cols">
  <div class="col"><h3>Befunde für dieses Modell an dieser Stelle</h3><ul>
    <li>Die gemeinsamen Gruppen von J-Space und U-Space sind die Gruppen des Zustands.</li>
    <li>Der generische U-Space ist zu grob. Von seinen vier Achsen misst nur „widersprüchliche Belege“ nachweislich, was sie heißt.</li>
    <li>U++ trennt „Verhalten gezeigt“ von „nicht gezeigt“; ein großer Teil davon ist eine gemeinsame Richtung.</li>
    <li>Die Anzeige eines harten SVM und die Reaktion des Netzes lassen sich getrennt verändern, an der Ausgabe in vier Paaren und in den nachfolgenden Neuronen in sieben von acht.</li>
    <li>Die Aktivierungen sind in der Neuronen-Basis konzentriert. In vier Paaren bewegen 300 bis 1.000 gezielt gewählte Neuronen die Ausgabe (dieser Teil ist nicht unabhängig nachgerechnet).</li>
  </ul></div>
  <div class="col"><h3>Nicht gesichert</h3><ul>
    <li>Ein Vorzug der Neuronen über die Konzentration hinaus, eine wirksame Verdrahtung, eine Überlegenheit gegenüber SAE-Einheiten.</li>
    <li>Ob U++ das Verhalten erkennt oder nur eine ungewohnte Antwort.</li>
    <li>Ob Neuronen „deutbar“ sind, wurde nicht gemessen. Die Neuronen-Befunde betreffen Konzentration, Geometrie und Wirkung.</li>
  </ul></div>
  <div class="col"><h3>Grenzen</h3><ul>
    <li>Basismodell, Fragenfamilien, erstes Antwort-Token, ein Eingriffsort (Block 21, letztes Token der Eingabe).</li>
    <li>Acht von Hand gewählte, teils überlappende Paare und eine Aufteilung. Die Ausgabe-Messung ist in vier davon ausgewertet.</li>
    <li>SAE und Lens sind am letzten Token der Eingabe schwächer als auf gewöhnlichem Text.</li>
    <li>Blinde Tests: 4 bis 14 Einheiten je Art, 16 Fragen je Einheit.</li>
  </ul></div>
</div>
<div class="next"><h3>Nächster Schritt</h3>
<p>Als Nächstes derselbe Versuch wie in Abschnitt 4 mit feinjustierten Modellen: Die beiden Klassen sind dann das feinjustierte Modell und das Basismodell auf denselben Fragen, die Richtung ist die mittlere Verschiebung des Zustands durch das Fine-Tuning. Dann lässt sich prüfen, ob diese Verschiebung im Basismodell das Verhalten auslöst und ob eine Trennlinie zwischen den Modellen mehr anzeigt, als das Netz daraus macht. Das braucht eine GPU.</p></div>
</section>""")

    # ---------------------------------------------------------------- Quellen
    P_.append("""<footer class="foot">
<p>Zahlen und Skripte: <span class="id">results/ju_space/README.md</span> und <span class="id">analysis/ju_space/</span> im DSSE-Repository. Alle Zahlen stammen aus einem Lauf auf 4 CPU-Kernen. Vier unabhängige Prüfdurchgänge haben Code und Deutung zu den Gruppierungen (Abschnitt 1), zum Trennlinien-Versuch (Abschnitt 4) und zu den Neuronen (Abschnitt 6: Ausdünnen, Verdrahtung, Trennen, Gegenprobe) nachgerechnet. Für die Abschnitte 2, 3 und 5 sowie für die Wirkung und das Benennen der Neuronen liegt kein eigener Prüfbericht vor.</p>
<p>Bausteine: Daten Transluce DSSE · J-Lens <span class="id">neuronpedia/jacobian-lens</span> · U-Space nach Braun et al., <a href="https://arxiv.org/abs/2610.09087" target="_blank" rel="noopener">arXiv 2610.09087</a> · Angriff nach Biggio et al., <a href="https://arxiv.org/abs/1708.06131" target="_blank" rel="noopener">arXiv 1708.06131</a> · SAE <span class="id">adamkarvonen/qwen3-1.7b-saes</span>. Anlass für Abschnitt 6: Goodfire, <a href="https://www.goodfire.com/research/neurons-as-an-interpretable-basis" target="_blank" rel="noopener">„Neurons as an Interpretable Basis“</a>; daraus ist nichts übernommen.</p>
</footer>""")
    return "".join(P_)


def polish(h):
    """Geschuetzte Leerzeichen vor %, kein Umbruch in Spannen und in Begriffen wie U-Space (nur im sichtbaren Text, nicht in Attributen)."""
    h = re.sub(r"(\d) %", "\\1\u00a0%", h)
    parts = re.split(r"(<[^>]+>)", h)
    term = re.compile(r"(?<![\w-])([A-Z]-[A-Za-zÄÖÜäöü]+(?:\+\+)?|[UJ]\+\+|\d+(?:,\d+)?[–/]\d+(?:,\d+)?(?:\u00a0%)?|5\.–95\.)")
    for i in range(0, len(parts), 2):
        parts[i] = term.sub(r'<span class="nw">\1</span>', parts[i])
    return "".join(parts)


def page(standalone=False):
    head = ('<title>J-Space und U-Space auf DSSE</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">\n'
            f"<style>{CSS}</style>\n")
    body = f'<div class="page" lang="de">{polish(build())}</div>\n<script>{JS}</script>\n'
    if not standalone:
        return head + body
    reset = (":root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}"
             "body{margin:0;font:14px system-ui,sans-serif}img{max-width:100%}[hidden]{display:none!important}")
    return ('<!doctype html>\n<html lang="de">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            f"<style>{reset}</style>\n{head}</head>\n<body>\n{body}</body>\n</html>\n")


if __name__ == "__main__":
    out = sys.argv[1]
    open(out, "w", encoding="utf-8").write(page("--standalone" in sys.argv))
    print("geschrieben:", out)
