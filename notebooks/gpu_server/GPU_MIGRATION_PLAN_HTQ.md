# HTQ: Restlaufzeit und Vorbereitung eines GPU-Wechsels

**Verworfen durch Nutzerentscheidung am 16.11.2026:** Es bleibt bei Colab/A100.
Diese Datei ist nur noch die historische Vergleichsrechnung; sie ist kein aktueller Miet- oder Migrationsauftrag.
Es wurde keine externe GPU-Instanz gebucht. Der aktuelle Automatisierungsweg verwendet eine einzelne Colab-Laufzeit.

Stand: 16. November 2026. Dies ist eine Betriebs- und Kostenplanung, keine Änderung des eingefrorenen Messplans.
Keine Instanz wurde gemietet, kein Serverlauf gestartet und keine Änderung am A100-Hardwaretor SG2 vorgenommen.

## Ausgangspunkt und belastbare Grenzen

Der neue HTQ-START/K_S1 ist abgeschlossen: 7.736 s, rund 2 h 9 min. Davon entfielen 6.096 s auf k3e.
B2 war beim letzten vorliegenden Protokoll vor der Berechnung an der Colab-Drive-Anmeldung gescheitert.
Der kleine operative Satz H_tq ist eine Neuronenmenge; Fragen, Stichproben und die deskriptive H+-Linse bleiben erhalten.

Für die verbleibenden Hauptsitzungen ergibt die alte P0b-Kostenbasis ungefähr **18–20 A100-Stunden**.
Das ist eine optimistische Planbasis, keine gemessene Prognose oder Obergrenze: k3e brauchte im neuen START rund das
8,6-Fache seines alten Ansatzes einschließlich Sicherheitsmarge. Diesen Faktor auf alle übrigen Messungen anzuwenden
wäre ebenso unbegründet wie ihn zu ignorieren. Die übrigen START-Schritte dauerten zusammen nur etwa 27 Minuten.

Enthalten sind die offenen K-/L-Hauptsitzungen, restliche R-Arbeit und D_C sowie deren Berichte. K_S4 wird vollständig
neu gerechnet; sein altes H+-k8b ist kein wiederverwendbarer HTQ-Schritt. D_C ist teilweise erledigt; seine tatsächlichen
Marker müssen beim Import geprüft werden. Separate spätere Tier-2-Sitzungen können nach alter Kostenbasis weitere
bis zu etwa **8,8 A100-Stunden** benötigen. L-CAL auf CPU kommt separat hinzu und muss vor ANA vorliegen und
ordnungsgemäß übernommen werden. Eine GPU-Abschlusszeit allein ist deshalb kein Termin für den fertigen Gesamtbericht.
Mit allen noch separaten finanzierten Zusatzläufen beträgt diese alte Gesamtbasis somit etwa **27–29 A100-Stunden**,
je nachdem, welche bedingten R-Läufe tatsächlich ausgelöst werden. Die Beschleunigungstabelle unten betrifft den Hauptlauf.

## Was tatsächlich parallel ist

Nach dem abgeschlossenen K_S1 können unabhängig voneinander starten:

- K_S2a → K_S2b; K_S2a liefert außerdem einen Eingang für L_B.
- K_S4; danach kann K_S5 starten, sobald auch K_S2a und die vorhandenen R-Eingänge verfügbar sind.
- L_A → L_B, wobei L_B zusätzlich K_S2a und den bestehenden GLCE-Eingang braucht.
- Die noch offenen R-Sitzungen und der Rest von D_C, soweit ihre vorhandenen Eingänge vollständig sind.

**K_S4 → L_A ist nur die bisherige B1-Notebook-Reihenfolge, keine fachliche Abhängigkeit.**
Ein Scheduler auf Sitzungsebene kann L_A früher starten. Eine zweite GPU reduziert Wartezeit merklich; bei drei oder
vier GPUs begrenzen insbesondere L_A → L_B und abschließende Berichte den Nutzen. Ein Modell über mehrere GPUs zu
verteilen ist für diese kleinen Modelle und ihre Hooks nicht der erste Ansatz. Eine GPU pro unabhängiger Sitzung
erhält die bestehenden Messschleifen und Stichproben.

Die reproduzierbare Rechnung steht in [htq_timing_scenario.json](htq_timing_scenario.json):
18,699 h seriell, 9,457 h auf zwei GPUs und 7,196 h auf drei oder vier GPUs bei gezielter Sitzungsverteilung.
Ganze bestehende Slots ergeben dagegen auch mit mehr GPUs etwa 9,671 h, weil B1 dann seriell bleibt.
Diese Zahlen zeigen die Abhängigkeiten; ihre Nachkommastellen sind keine zeitliche Prognosegenauigkeit.
Die spätere tatsächliche Scheduler-Reihenfolge und Import-/Publish-Kosten können davon abweichen.

## Rechenbeispiel für die Beschleunigung

Alle Zahlen der folgenden Tabelle sind **Szenarien für genau 20 h A100-Restarbeit**, keine auf H100/B200/B300 gemessenen
Laufzeiten. Setup, Datentransfer, Modell-Downloads, ein Hardwarevergleich und separate CPU-Kalibrierung sind nicht
enthalten. Dafür zunächst zusätzlich 0,5–1,5 h als organisatorische Reserve einplanen; große Transfers können länger dauern.

Verwendete ungemessene Faktoren: H100 PCIe 1,3–2,0; H100 SXM/H200 1,5–2,5; B200/B300 1,8–3,0 gegenüber A100.
Zusätzlicher Parallelfaktor bei zwei GPUs 1,6–1,9, bei vier GPUs 2,1–2,6. Diese Annahmen sind keine statistischen
Konfidenzintervalle; CPU-/Hook-/I/O-Anteile können auch geringere Beschleunigung verursachen.
Formel: `Zeit = 20 h / Hardwarefaktor / Parallelfaktor`.

| Konfiguration | Rechenzeit im Szenario, gerundet | Reine GPU-Miete für diese Zeit, ungefähr |
|---|---:|---:|
| 1× A100 | 20 h | 28 $ bei 1,38 $/h |
| 2× A100 | 11–13 h | 30–36 $ bei zwei Einzelinstanzen à 1,38 $/h |
| 1× H100 SXM | 8–14 h | 40–69 $ bei 4,94 $/h |
| 1× H200 SXM | 8–14 h | 48–83 $ bei 5,95 $/h |
| 2× H100 PCIe | 6–10 h | 34–56 $ bei 5,58 $/h für den ganzen Zweier-Server |
| 1× B200 | 7–12 h | 63–108 $ bei 8,99 $/h |
| 2× B200 | 4–7 h | 72–126 $ bei zwei Einzelinstanzen à 8,99 $/h |
| 1× B300 | 7–12 h | 71–122 $ bei 10,11 $/h |
| 2× B300 | 4–7 h | 81–142 $ bei zwei Einzelinstanzen à 10,11 $/h |
| 4× B200 | 3–6 h | 144–287 $ beim gelisteten Vierer-Server für 47,82 $/h |

Die Mietbeträge unterstellen, dass alle genannten GPUs während der gesamten angegebenen Zeit gemietet bleiben.
Frühes Freigeben fertiger Einzelinstanzen spart Geld; Setup, Prüfung, Speicherung und etwaige weitere Gebühren kosten
zusätzlich. Zweimal so viel tatsächliche A100-Arbeit verdoppelt in diesem vereinfachten Modell auch Zeit und Miete.

## Warum B300 hier kein sicherer Sprung gegenüber B200 ist

Der vorhandene Code schreibt Float32, TF32 aus und Matmul-Präzision `highest` vor. Zudem wird der effiziente
SDPA-Backend ausdrücklich erzwungen; ein stiller Wechsel auf einen anderen Backend oder auf BF16/FP8/FP4 ist
keine unveränderte Fortsetzung dieses Versuchs. NVIDIA nennt für HGX B200 und B300 jeweils 600 TFLOPS FP32 für acht
GPUs und bis zu 8 TB/s Speicherbandbreite pro GPU. B300 hat mehr Speicher, den diese kleinen Modelle nicht allein
wegen ihrer Gewichte benötigen. Verbesserungen einzelner Attention-Kernel sind möglich, aber hier nicht gemessen.
Deshalb wird **kein zusätzlicher B300-Speedup** in die Rechnung eingesetzt.

Quellen: [NVIDIA HGX](https://www.nvidia.com/en-us/data-center/hgx/),
[NVIDIA H200/B200/B300-Spezifikationen](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html).

## Aktuelle Angebote und deren Grenzen

Am 16.11.2026 um 22:32 UTC wurden alle Seiten des öffentlichen GPU.ai-Katalogs gelesen: 246 Angebotszeilen,
einschließlich nicht verfügbarer Angebote. Preise sind USD; Verfügbarkeit ist nur eine Momentaufnahme.
Der Auszug liegt in [htq_gpuai_price_snapshot.json](htq_gpuai_price_snapshot.json).

- B200: 1 GPU in EU East, 8,99 $/h, vier verfügbare Einheiten; 32 CPU-Kerne, 323 GB RAM, 100 GB Instanzdisk.
- B300: 1 GPU in EU West, 10,11 $/h, vier verfügbare Einheiten; 32 CPU-Kerne, 251 GB RAM, 100 GB Instanzdisk.
- 2× H100 PCIe: eine VM in US Central für zusammen 5,58 $/h, fünf verfügbare Einheiten; 40 CPU-Kerne,
  256 GB RAM und 2,5 TB Disk. H100 PCIe und H100 SXM sind unterschiedliche Angebote.
- Ein verfügbarer 2×-B200- oder 2×-B300-Server war in diesem Snapshot nicht gelistet. Die Zweier-B-Szenarien oben
  meinen **zwei Einzelinstanzen**, mit zusätzlichen Datenübergaben zwischen den Maschinen.
- 4× B200: 47,82 $/h für den gesamten Server; 8× B200: 89,90 $/h. Eine Acht-GPU-Miete ist für diese Rest-DAG
  derzeit nicht sinnvoll begründet.
- Die gefundenen B200/B300-Angebote nennen ein Basisimage mit CUDA 12.8, der H100-Server CUDA 12.4. Ein
  Anbieterlabel „certified“ ersetzt den Test unserer konkreten Torch-/Attention-/Hook-Kombination nicht.

Die API bezeichnet `price_per_hour` als Preis **der ganzen Zeile einschließlich `gpu_count`**. Diesen Betrag nicht
erneut mit der GPU-Anzahl multiplizieren. Konfigurierbare Disk kann einen separaten Tarif haben; die Kostenrechnung
oben ist eine reine GPU-/Instanzmietrechnung.

Quellen: [GPU.ai Preise](https://gpu.ai/pricing),
[öffentliche Preis-API](https://api.gpu.ai/v1/pricing?include_unavailable=true&limit=200),
[GPU.ai Instanz- und Preisfelder](https://gpu.ai/docs/guides/gpu-instances).

## Vorbereitung und Freigabe eines Hardwarewechsels

Der aktuelle Runner akzeptiert in SG2 ausdrücklich nur A100/Compute Capability 8.0 mit mindestens 39 GB.
**H100/H200/B200/B300 sind damit heute keine direkt ausführbare Fortsetzung**, auch wenn das Mietangebot verfügbar ist.
Die Vorbereitung erhält dieses Tor. Eine neue Hardwarepolitik muss als gesonderte, nachvollziehbare Betriebsänderung
umgesetzt werden; `MOCK=1`, ein CPU-Schalter oder ein umbenannter GPU-Name sind kein Ersatz.

Vor einem solchen Wechsel:

1. Einen abgeschlossenen, unveränderlichen Snapshot der benötigten A-/B-Ergebnisse und R_S0-Exporte herstellen.
   Alle `.pt`-Dateien einschließlich Hidden States erhalten. Import mit SHA-256 prüfen; historische Seed-Dateihashes
   nicht mit inzwischen fortgeschriebenen Ergebnisdateien verwechseln. Die Colab-Pfade in Seed-Markierungen müssen
   ausdrücklich und mit erhaltener Originalhistorie auf lokale Pfade übertragen werden.
2. Lokal auf NVMe rechnen. Google Drive dient dem Transport und der späteren Sicherung; `google.colab.drive.mount`
   gehört nicht in den Serverlauf. Ein Download aus dem Drive-Webbrowser kann weiterhin möglich sein, obwohl die
   Colab-Mount-Anmeldung defekt ist. Ob er beim Nutzer funktioniert, wurde noch nicht geprüft.
3. Pro Worker eigene A-/B-Snapshots, ROOT, WORK und Logs; ausschließlich ein Koordinator veröffentlicht fertige
   Sitzungen. Der bestehende Runner sperrt weder parallele Sitzungsausführungen noch alle Kopier-/Exportmanifeste.
   Vorhandene Colab-Schreiber auf denselben Daten müssen beendet sein, bevor ein Snapshot als Ausgangspunkt gilt.
4. GPU, Compute Capability, Treiber, Torch/CUDA, Transformers und Paketstand protokollieren. Der erfolgreiche START
   verwendete Torch 2.14.1+cu130 und Transformers 5.22.2. Keine unkontrollierten Paket-Upgrades aus dem älteren
   `gpu_server/setup.sh` übernehmen.
5. Auf der Zielhardware den echten FP32-Forward mit erzwungenem EFFICIENT_ATTENTION und Hooks prüfen, einschließlich
   benötigter langer Kontexte. Danach kurze repräsentative K-Linsen-, L-Decode- und gegebenenfalls D-Messungen mit
   Warm-up und CUDA-Synchronisierung zeitlich messen. Ein BF16-Matmul-Test oder `cuda.is_available()` genügt nicht.
6. Einen festgelegten kleinen Vergleichssatz mit A100-Eingaben/Referenzen vor der Ergebnisinterpretation prüfen und
   Differenzen dokumentieren. Keine Bitgleichheit über GPU-Architekturen behaupten. Erst danach Hardwarepolitik und
   Budget für den vollständigen Lauf konkretisieren.
   Der bereits gespeicherte K3a-Mitschnitt enthält hierfür FP32-Referenzzustände und genaue Prompttokens von 256
   A100-Anfragen. Er kann einen vorher festgelegten Prefill-Vergleich liefern, ersetzt aber keine K-/L-Laufzeitmessung
   und keinen Test langer Kontexte. START muss dafür nicht erneut vollständig gerechnet werden.
7. L-CAL/ANA getrennt auf CPU planen. Ergebnisse regelmäßig extern sichern und Instanzen nach bestätigter Sicherung
   freigeben; lokale Instanzdisk ist kein dauerhaftes Archiv.

NVIDIA führt H100/H200 als CC 9.0, B200 als 10.0 und B300 als 10.3. CUDA 12.9 führt native `sm_103`-Compilerziele ein;
ältere Binär-/PTX-Kompatibilität ist kein Nachweis, dass alle benötigten Attention-Kernel funktionieren.
Quellen: [GPU-Architekturen](https://developer.nvidia.com/cuda/gpus),
[CUDA-12.9-Release-Notes](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html#general-cuda),
[PyTorch numerische Genauigkeit](https://docs.pytorch.org/docs/2.15/notes/numerical_accuracy.html).

## Empfehlung

Für den kleinsten Wechsel zuerst zwei A100-Sitzungen parallel betreiben: dieselbe zugelassene GPU-Klasse und ein
großer Teil des Parallelgewinns. Für einen externen, neu zu prüfenden Server ist **2× H100 PCIe** das interessante
Kostenangebot. **2× B200** ist die Geschwindigkeitsoption, wenn ein kurzer echter Projektbenchmark den Mehrpreis
rechtfertigt und der Transfer zwischen zwei Maschinen vorbereitet ist. B300 nur wählen, wenn der eigene Benchmark
oder ein günstigerer aktueller Tarif dafür spricht. Vier oder acht Blackwell-GPUs nicht allein aus theoretischen
FLOPS ableiten.

Die konkrete lokale Vorbereitung und ihre Ausführungsgrenzen stehen in [README_HTQ.md](README_HTQ.md).
