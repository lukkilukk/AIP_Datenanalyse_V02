import argparse
import traceback
from pathlib import Path

from rolling_report.pipeline import QualityReportPipeline

DEFAULT_INPUT_DIR = Path('analysedaten') / 'eingang'
DEFAULT_OUTPUT_DIR = Path('analysedaten') / 'output'


def resolve_input_file(raw_value: str) -> Path:
    candidates = [
        Path(raw_value),
        DEFAULT_INPUT_DIR / raw_value,
        Path('analysedaten') / raw_value,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(raw_value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Analysiert eine Walzwerk-Messdatei und erzeugt Report, Kennzahlen und Prediction-Output.'
    )
    parser.add_argument('file', help='Dateiname oder Pfad zur Messdatei, z. B. Montag.txt')
    parser.add_argument(
        '--output',
        default=str(DEFAULT_OUTPUT_DIR),
        help='Ausgabeverzeichnis (Standard: analysedaten/output)',
    )
    parser.add_argument(
        '--startup-seconds',
        type=int,
        default=60,
        help='Anlauf-Fenster fuer Coil-Features in Sekunden (Standard: 60)',
    )
    parser.add_argument(
        '--rf-estimators',
        type=int,
        default=300,
        help='Anzahl Baeume fuer den Random Forest (Standard: 300)',
    )
    parser.add_argument(
        '--rf-max-depth',
        type=int,
        default=6,
        help='Maximale Baumtiefe fuer den Random Forest (Standard: 6)',
    )
    args = parser.parse_args()

    file_path = resolve_input_file(args.file)
    if not file_path.exists():
        print(f'Fehler: Datei nicht gefunden: {args.file}')
        print(f'Tipp: Lege die Datei in {DEFAULT_INPUT_DIR} oder uebergib einen vollen Pfad.')
        return 1

    try:
        pipeline = QualityReportPipeline(
            startup_seconds=args.startup_seconds,
            rf_estimators=args.rf_estimators,
            rf_max_depth=args.rf_max_depth,
        )
        result = pipeline.run(file_path=file_path, output_dir=Path(args.output))
    except KeyboardInterrupt:
        print('\n[ABBRUCH] Analyse durch Benutzer abgebrochen.')
        return 130
    except Exception as exc:
        print(f'\n[FEHLER] Analyse fehlgeschlagen: {exc}')
        traceback.print_exc()
        return 1

    report_path = result['report_path']
    management_report_path = result['management_report_path']
    print('\n[ERFOLGREICH] Analyse erfolgreich abgeschlossen.')
    print(f'Input: {file_path}')
    print(f'Report: {report_path}')
    print(f'Gesamtreport HTML: {management_report_path}')
    print(f"Coils: {result['summary']['n_coils']}")
    print(f"Produktionszyklen: {result['summary']['n_cycles']}")
    print(f"Schlechte Coils: {result['summary']['n_bad_coils']}")
    print(f"Auffaellige Coils: {result['summary'].get('n_alert_coils', 0)}")
    if result['summary']['rf_accuracy_mean'] is not None:
        mean = result['summary']['rf_accuracy_mean']
        std = result['summary']['rf_accuracy_std']
        print(f'RF Accuracy: {mean:.1%} +/- {std:.1%}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
