# AIP Datenanalyse

Eine schlanke Pipeline fuer Walzwerk-Messdateien mit klarer Trennung zwischen Code und Analysedaten.

## Ordnerstruktur

- `rolling_report/`
  hier liegt nur der Code
- `analysedaten/eingang/`
  hier kommen die Messdateien rein
- `analysedaten/output/`
  hier landen Reports, CSVs und Plots

## Schnellstart

Per Doppelklick auf `run.cmd` oeffnet sich jetzt ein Dateiauswahl-Fenster. Standardmaessig startet es im Ordner `analysedaten/eingang/`.

Alternativ kannst du im Projektordner weiter direkt ueber die Shell starten:

```powershell
.\run.cmd DEINE_DATEI.txt
```

Beispiele:

```powershell
.\run.cmd Montag.txt
.\run.cmd Dienstag.txt
.\run.cmd coil_2026_03_24.txt
```

Der Wrapper startet automatisch mit der vorgesehenen Conda-Umgebung:
`C:\Users\luk3b\anaconda3\envs\AIP_Assistant`

## Was Du Beim Start Siehst

`run.cmd` zeigt beim Start sofort eine klare Meldung an:

- dass die Analyse gestartet wurde
- welche Datei verarbeitet wird
- dass du mit `STRG+C` abbrechen kannst
- dass das Fenster am Ende offen bleibt

Waehrend des Laufs siehst du Fortschrittsschritte wie:

- `[1/7] Lade Messdatei`
- `[2/7] Erkenne Betriebsphasen und Produktionszyklen`
- `[5/7] Trainiere Random Forest und berechne Praediktoren`
- `[7/7] Erzeuge Reports fuer Team, Leitung und Fahrer`

Am Ende bleibt das Fenster offen, bis du eine Taste drueckst.

Die Abschlussmeldung ist eindeutig:

- `[ERFOLGREICH] Analyse erfolgreich abgeschlossen.`
- `[ABBRUCH] Analyse wurde vom Benutzer abgebrochen.`
- `[FEHLER] Analyse ist mit Fehler beendet.`

## Startmoeglichkeiten

- Doppelklick auf `run.cmd` und Datei im Auswahlfenster waehlen
- Shell-Start mit `.\run.cmd DATEI.txt`
- Drag-and-drop einer Datei auf `run.cmd`

## Drag-And-Drop

Du kannst eine Messdatei auch einfach auf [run.cmd](c:/Projekte/AIP_Datenanalyse/run.cmd) ziehen.
Dann wird sie direkt analysiert, auch wenn sie nicht in `analysedaten/eingang/` liegt.

## Manueller Start Mit Conda

```powershell
conda activate AIP_Assistant
python run_analysis.py DEINE_DATEI.txt
```

Oder direkt mit dem Python der Umgebung:

```powershell
C:\Users\luk3b\anaconda3\envs\AIP_Assistant\python.exe run_analysis.py DEINE_DATEI.txt
```

## Wichtige Pfade

- Input-Ordner: `analysedaten/eingang/`
- Output-Ordner: `analysedaten/output/`
- Hauptreport: `analysedaten/output/reports/quality_report.md`
- Gesamtreport HTML: `analysedaten/output/reports/gesamtreport_produktion.html`
- Summary: `analysedaten/output/reports/summary.json`

## Output

Die Pipeline schreibt alles nach `analysedaten/output/`:

- `reports/quality_report.md`
- `reports/gesamtreport_produktion.html`
- `reports/summary.json`
- `reports/*.csv`
- `prediction/*.csv`
- `plots/*.html`
