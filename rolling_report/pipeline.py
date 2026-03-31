from __future__ import annotations

from pathlib import Path

from .analysis import (
    add_phase_information,
    build_anomaly_summary,
    build_planarity_summary,
    build_recommendations,
    build_thickness_summary,
    detect_coils,
    detect_cycles,
    summarize_phases,
)
from .config import DEFAULT_STARTUP_SECONDS
from .io import load_measurement_file
from .prediction import build_feature_matrix, run_prediction
from .reporting import (
    create_output_dirs,
    export_json,
    export_table,
    plot_phase_summary,
    plot_planarity_profile,
    plot_thickness_summary,
    write_management_html_report,
    write_markdown_report,
)


class QualityReportPipeline:
    def __init__(self, startup_seconds: int = DEFAULT_STARTUP_SECONDS, rf_estimators: int = 300, rf_max_depth: int = 6):
        self.startup_seconds = startup_seconds
        self.rf_estimators = rf_estimators
        self.rf_max_depth = rf_max_depth

    def _status(self, step: int, total: int, text: str) -> None:
        print(f'[{step}/{total}] {text}')

    def run(self, file_path: Path, output_dir: Path) -> dict[str, object]:
        total_steps = 7
        directories = create_output_dirs(output_dir)

        self._status(1, total_steps, 'Lade Messdatei')
        df, meta = load_measurement_file(file_path)
        print(f'      Datei: {meta.file_path.name} | Zeilen: {meta.n_rows} | Spalten: {meta.n_columns} | Takt: {meta.step_seconds}s')

        self._status(2, total_steps, 'Erkenne Betriebsphasen und Produktionszyklen')
        enriched = add_phase_information(df, step_seconds=meta.step_seconds)
        phase_summary = summarize_phases(enriched, meta.step_seconds)
        cycle_summary = detect_cycles(enriched, meta.time_column, meta.step_seconds)

        self._status(3, total_steps, 'Erkenne Coils und berechne Qualitaetskennzahlen')
        coil_summary = detect_coils(enriched, meta.time_column, meta.step_seconds)
        thickness_summary = build_thickness_summary(enriched)
        planarity_profile, planarity_metrics = build_planarity_summary(enriched)
        anomaly_summary = build_anomaly_summary(enriched)
        print(f'      Coils erkannt: {len(coil_summary)} | Produktionszyklen: {len(cycle_summary)}')

        self._status(4, total_steps, 'Erzeuge Feature-Matrix fuer Prediction')
        feature_matrix = build_feature_matrix(
            enriched,
            coil_summary,
            step_seconds=meta.step_seconds,
            startup_seconds=self.startup_seconds,
            time_column=meta.time_column,
        )
        print(f'      Feature-Matrix: {len(feature_matrix)} Coils x {len(feature_matrix.columns) if not feature_matrix.empty else 0} Spalten')

        self._status(5, total_steps, 'Trainiere Random Forest und berechne Praediktoren')
        prediction_artifacts = run_prediction(
            feature_matrix,
            prediction_dir=directories['prediction'],
            rf_params={
                'n_estimators': self.rf_estimators,
                'max_depth': self.rf_max_depth,
            },
        )

        self._status(6, total_steps, 'Leite Empfehlungen ab und schreibe CSV/Plots')
        recommendations = build_recommendations(
            thickness_summary=thickness_summary,
            prediction_summary={
                'quality_by_stitch': prediction_artifacts.quality_by_stitch,
                'top_rf_features': prediction_artifacts.top_rf_features,
            },
            planarity_metrics=planarity_metrics,
            anomaly_summary=anomaly_summary,
        )

        export_table(phase_summary, directories['reports'] / 'phase_summary.csv')
        export_table(cycle_summary, directories['reports'] / 'cycle_summary.csv')
        export_table(coil_summary, directories['reports'] / 'coil_summary.csv')
        export_table(thickness_summary, directories['reports'] / 'thickness_summary.csv')
        export_table(planarity_profile, directories['reports'] / 'planarity_profile.csv')
        export_table(anomaly_summary, directories['reports'] / 'anomaly_summary.csv')

        plot_phase_summary(phase_summary, directories['plots'])
        plot_thickness_summary(thickness_summary, directories['plots'])
        plot_planarity_profile(planarity_profile, directories['plots'])

        quality_series = prediction_artifacts.feature_matrix.get('quality')
        bad_coils = int((quality_series == 'schlecht').sum()) if quality_series is not None else 0
        alert_coils = int((quality_series == 'auffaellig').sum()) if quality_series is not None else 0
        summary = {
            'n_coils': int(len(coil_summary)),
            'n_bad_coils': bad_coils,
            'n_alert_coils': alert_coils,
            'n_cycles': int(len(cycle_summary)),
            'rf_accuracy_mean': prediction_artifacts.accuracy_mean,
            'rf_accuracy_std': prediction_artifacts.accuracy_std,
        }
        export_json(summary, directories['reports'] / 'summary.json')

        self._status(7, total_steps, 'Erzeuge Reports fuer Team, Leitung und Fahrer')
        report_path = directories['reports'] / 'quality_report.md'
        management_report_path = directories['reports'] / 'gesamtreport_produktion.html'

        write_markdown_report(
            report_path=report_path,
            meta=meta,
            summary=summary,
            thickness_summary=thickness_summary,
            anomaly_summary=anomaly_summary,
            planarity_metrics=planarity_metrics,
            prediction_artifacts=prediction_artifacts,
            recommendations=recommendations,
        )
        write_management_html_report(
            report_path=management_report_path,
            meta=meta,
            summary=summary,
            phase_summary=phase_summary,
            cycle_summary=cycle_summary,
            thickness_summary=thickness_summary,
            anomaly_summary=anomaly_summary,
            planarity_profile=planarity_profile,
            planarity_metrics=planarity_metrics,
            prediction_artifacts=prediction_artifacts,
            recommendations=recommendations,
        )

        return {
            'report_path': report_path,
            'management_report_path': management_report_path,
            'summary': summary,
            'phase_summary': phase_summary,
            'cycle_summary': cycle_summary,
            'coil_summary': coil_summary,
            'feature_matrix': prediction_artifacts.feature_matrix,
        }
