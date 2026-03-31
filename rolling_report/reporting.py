from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio

from .config import PLOTS_DIRNAME, PREDICTION_DIRNAME, REPORTS_DIRNAME


def create_output_dirs(base_dir: Path) -> dict[str, Path]:
    reports_dir = base_dir / REPORTS_DIRNAME
    prediction_dir = base_dir / PREDICTION_DIRNAME
    plots_dir = base_dir / PLOTS_DIRNAME
    for directory in (base_dir, reports_dir, prediction_dir, plots_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return {
        'base': base_dir,
        'reports': reports_dir,
        'prediction': prediction_dir,
        'plots': plots_dir,
    }



def export_table(df: pd.DataFrame, path: Path) -> None:
    if df is None or df.empty:
        return
    df.to_csv(path, sep=';', decimal=',', index=False)



def export_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')



def plot_phase_summary(phase_summary: pd.DataFrame, plots_dir: Path) -> None:
    if phase_summary.empty:
        return
    figure = px.bar(
        phase_summary,
        x='phase_name',
        y='duration_min',
        text='duration_min',
        title='Dauer pro Betriebsphase',
    )
    figure.write_html(plots_dir / 'phase_summary.html', include_plotlyjs='cdn')



def plot_thickness_summary(thickness_summary: pd.DataFrame, plots_dir: Path) -> None:
    if thickness_summary.empty:
        return
    figure = px.bar(
        thickness_summary,
        x='phase_name',
        y='std_pct',
        text='std_pct',
        title='Standardabweichung der Dickenabweichung je Phase',
    )
    figure.write_html(plots_dir / 'thickness_by_phase.html', include_plotlyjs='cdn')



def plot_planarity_profile(planarity_profile: pd.DataFrame, plots_dir: Path) -> None:
    if planarity_profile.empty:
        return
    figure = px.line(
        planarity_profile,
        x='zone',
        y='mean_value',
        markers=True,
        title='Mittleres Planheitsprofil ueber die Bandbreite',
    )
    figure.write_html(plots_dir / 'planarity_profile.html', include_plotlyjs='cdn')



def _format_metric(value: object, suffix: str = '') -> str:
    if value is None:
        return '-'
    if isinstance(value, (int, float)):
        return f'{value:.1f}{suffix}' if isinstance(value, float) else f'{value}{suffix}'
    return f'{value}{suffix}'



def _pretty_feature_name(name: str) -> str:
    text = str(name)
    replacements = [
        ('local_peak_', 'Lokales Stoerfenster | '),
        ('local_', 'Lokal | '),
        ('baseline_', 'Basiscoil | '),
        ('_startup_', ' | Anlauf | '),
        ('_stable_', ' | Stabil | '),
        ('_shift', ' Delta'),
        ('_mean', ' Mittelwert'),
        ('_std', ' Std'),
        ('_max', ' Max'),
        ('_min', ' Min'),
        ('_pct', ' %'),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    pretty_tokens = [
        ('edge_center', 'Rand Mitte'),
        ('left_right', 'Links Rechts'),
        ('shape_profile', 'Planheitsprofil'),
        ('force_imbalance', 'Differenzkraftanteil'),
        ('pressure_gap', 'Druckdifferenz'),
        ('tension_gap', 'Zugdifferenz'),
        ('speed_tracking_error', 'Geschwindigkeitsfehler'),
        ('bending_activity', 'Biegeregler Aktivitaet'),
        ('slant_activity', 'Schraeglageregler Aktivitaet'),
        ('quality_window_seconds', 'Hotspot Fenster s'),
        ('q95', 'Q95'),
        ('dvdt_activity', 'dVdt Aktivitaet'),
    ]
    for old, new in pretty_tokens:
        text = text.replace(old, new)
    text = text.replace('_', ' ')
    return text.strip()



def _prepare_table(df: pd.DataFrame, max_rows: int = 10) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    table = df.head(max_rows).copy()
    for column in table.columns:
        if pd.api.types.is_datetime64_any_dtype(table[column]):
            table[column] = table[column].astype(str)
        elif pd.api.types.is_numeric_dtype(table[column]):
            table[column] = table[column].map(lambda value: '' if pd.isna(value) else f'{value:.2f}')
        else:
            table[column] = table[column].astype(str)
    return table



def _table_html(df: pd.DataFrame, max_rows: int = 10) -> str:
    table = _prepare_table(df, max_rows=max_rows)
    if table.empty:
        return '<p class="empty-state">Keine Daten verfuegbar.</p>'
    return table.to_html(index=False, classes='data-table', border=0, escape=True)



def _figure_html(figure, include_js: bool) -> str:
    return pio.to_html(
        figure,
        full_html=False,
        include_plotlyjs='cdn' if include_js else False,
        config={'displaylogo': False, 'responsive': True},
    )



def _build_management_figures(
    phase_summary: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    thickness_summary: pd.DataFrame,
    anomaly_summary: pd.DataFrame,
    planarity_profile: pd.DataFrame,
    prediction_artifacts,
) -> list[str]:
    snippets: list[str] = []
    include_js = True

    if not phase_summary.empty:
        figure = px.bar(
            phase_summary,
            x='phase_name',
            y='duration_min',
            color='phase_name',
            text='duration_min',
            title='Betriebsphasen ueber den Tag',
        )
        snippets.append(_figure_html(figure, include_js))
        include_js = False

    if not thickness_summary.empty:
        figure = px.bar(
            thickness_summary,
            x='phase_name',
            y='std_pct',
            color='phase_name',
            text='std_pct',
            title='Dickenstreuung je Phase',
        )
        snippets.append(_figure_html(figure, include_js))
        include_js = False

    if not cycle_summary.empty:
        figure = px.bar(
            cycle_summary,
            x='cycle_id',
            y='duration_min',
            title='Produktionsdauer je Zyklus',
            text='duration_min',
        )
        snippets.append(_figure_html(figure, include_js))
        include_js = False

    quality_by_stitch = prediction_artifacts.quality_by_stitch
    if quality_by_stitch is not None and not quality_by_stitch.empty:
        figure = px.bar(
            quality_by_stitch,
            x='stitch',
            y='bad_rate_pct',
            text='bad_rate_pct',
            title='Fehlerquote pro Stich',
            color='bad_rate_pct',
            color_continuous_scale='Reds',
        )
        snippets.append(_figure_html(figure, include_js))
        include_js = False

    feature_matrix = prediction_artifacts.feature_matrix
    if feature_matrix is not None and not feature_matrix.empty and 'thickness_sigma_pct' in feature_matrix.columns:
        plot_df = feature_matrix.dropna(subset=['thickness_sigma_pct']).copy()
        if not plot_df.empty:
            hover_columns = [
                column
                for column in [
                    'duration_min',
                    'quality_threshold_pct',
                    'quality_score',
                    'quality_flags',
                    'max_abs_thickness_pct',
                    'share_abs_gt_2pct',
                    'local_peak_sigma_pct',
                    'local_edge_center_shift',
                    'quality_driver',
                    'quality_detail',
                    'quality_rule',
                ]
                if column in plot_df.columns
            ]
            figure = px.scatter(
                plot_df,
                x='coil_id',
                y='thickness_sigma_pct',
                color='quality',
                symbol='stitch',
                title='Qualitaetslage pro Coil',
                hover_data=hover_columns,
            )
            snippets.append(_figure_html(figure, include_js))
            include_js = False

    if not planarity_profile.empty:
        figure = px.line(
            planarity_profile,
            x='zone',
            y='mean_value',
            markers=True,
            title='Mittleres Planheitsprofil',
        )
        snippets.append(_figure_html(figure, include_js))
        include_js = False

    top_rf = prediction_artifacts.top_rf_features
    if top_rf is not None and not top_rf.empty:
        plot_df = top_rf.head(12).copy().iloc[::-1]
        plot_df['feature_pretty'] = plot_df['feature'].map(_pretty_feature_name)
        figure = px.bar(
            plot_df,
            x='importance',
            y='feature_pretty',
            orientation='h',
            title='Wichtigste Random-Forest-Treiber',
        )
        snippets.append(_figure_html(figure, include_js))
        include_js = False

    top_corr = prediction_artifacts.top_correlations
    if top_corr is not None and not top_corr.empty:
        plot_df = top_corr.head(12).copy().iloc[::-1]
        plot_df['feature_pretty'] = plot_df['feature'].map(_pretty_feature_name)
        figure = px.bar(
            plot_df,
            x='correlation',
            y='feature_pretty',
            orientation='h',
            color='correlation',
            title='Staerkste lineare Zusammenhaenge mit dem Qualitaetslabel',
            color_continuous_scale='RdBu',
            color_continuous_midpoint=0,
        )
        snippets.append(_figure_html(figure, include_js))
        include_js = False

    if not anomaly_summary.empty:
        plot_df = anomaly_summary.head(10).copy().iloc[::-1]
        figure = px.bar(
            plot_df,
            x='anomaly_count',
            y='signal',
            orientation='h',
            title='Robuste Ausreisser in Produktionsphase',
        )
        snippets.append(_figure_html(figure, include_js))

    return snippets



def _build_operator_focus(recommendations: list[str], prediction_artifacts, anomaly_summary: pd.DataFrame) -> list[str]:
    focus: list[str] = []

    quality_by_stitch = prediction_artifacts.quality_by_stitch
    if quality_by_stitch is not None and not quality_by_stitch.empty:
        worst = quality_by_stitch.sort_values('bad_rate_pct', ascending=False).iloc[0]
        if worst['bad_rate_pct'] > 0:
            focus.append(f'Stich {int(worst["stitch"])} besonders eng fahren und aktiv ueberwachen.')

    top_rf = prediction_artifacts.top_rf_features
    if top_rf is not None and not top_rf.empty:
        feature_names = top_rf['feature'].astype(str)
        if feature_names.str.contains('startup').sum() >= 2:
            focus.append('Die ersten 60 Sekunden eines Coils sind kritisch und sollten aktiv beobachtet werden.')
        if feature_names.str.contains('gcurves_delten_27|gcurves_delten_28|edge_center|left_right').any():
            focus.append('Bandrand und Rand-Mitte-Spruenge als Fruehwarnsignal beobachten.')
        if feature_names.str.contains('local_|_shift').any():
            focus.append('Lokale Stoerfenster innerhalb eines Coil getrennt pruefen; Vollcoil-Mittelwerte koennen sie verwischen.')

    if not anomaly_summary.empty:
        top_signal = anomaly_summary.iloc[0]
        focus.append(f"Signal {top_signal['signal']} bei Auffaelligkeiten zuerst technisch pruefen.")

    for item in recommendations:
        if item not in focus:
            focus.append(item)

    return focus[:6]



def write_markdown_report(
    report_path: Path,
    meta,
    summary: dict[str, object],
    thickness_summary: pd.DataFrame,
    anomaly_summary: pd.DataFrame,
    planarity_metrics: dict[str, float] | None,
    prediction_artifacts,
    recommendations: list[str],
) -> None:
    top_rf_lines: list[str] = []
    if not prediction_artifacts.top_rf_features.empty:
        for row in prediction_artifacts.top_rf_features.head(5).itertuples(index=False):
            top_rf_lines.append(f"- `{row.feature}`: {row.importance:.4f}")
    else:
        top_rf_lines.append('- Kein belastbares Random-Forest-Modell moeglich.')

    top_corr_lines: list[str] = []
    if not prediction_artifacts.top_correlations.empty:
        for row in prediction_artifacts.top_correlations.head(5).itertuples(index=False):
            top_corr_lines.append(f"- `{row.feature}`: r={row.correlation:.3f}")
    else:
        top_corr_lines.append('- Keine belastbaren Korrelationen verfuegbar.')

    recommendation_lines = [f'- {item}' for item in recommendations]
    anomaly_lines = []
    if not anomaly_summary.empty:
        for row in anomaly_summary.head(5).itertuples(index=False):
            anomaly_lines.append(f"- `{row.signal}`: {row.anomaly_count} Ausreisser ({row.anomaly_share_pct:.2f}%)")
    else:
        anomaly_lines.append('- Keine robusten Ausreisser in den Standardsignalen erkannt.')

    stitch_lines = []
    quality_by_stitch = prediction_artifacts.quality_by_stitch
    if not quality_by_stitch.empty:
        for row in quality_by_stitch.itertuples(index=False):
            stitch_lines.append(
                f"- Stich {int(row.stitch)}: {int(row.bad_coils)} schlechte Coils von {int(row.total_coils)} ({row.bad_rate_pct:.1f}%)"
            )
    else:
        stitch_lines.append('- Keine Stich-Auswertung verfuegbar.')

    thickness_lines = []
    if not thickness_summary.empty:
        for row in thickness_summary.itertuples(index=False):
            thickness_lines.append(
                f"- {row.phase_name}: Mittel {row.mean_pct:.3f}%, Std {row.std_pct:.3f}%, |Abw.|>2%: {int(row.count_abs_gt_2pct)}"
            )
    else:
        thickness_lines.append('- Keine Dickenabweichungsanalyse moeglich.')

    planarity_section = '- Keine Shapemeter-Zonen gefunden.'
    if planarity_metrics is not None:
        planarity_section = (
            f"- Mitte: {planarity_metrics['center_mean']:.3f}\n"
            f"- Rand: {planarity_metrics['edge_mean']:.3f}\n"
            f"- Rand minus Mitte: {planarity_metrics['edge_minus_center']:.3f}\n"
            f"- Volatilste Zone: {int(planarity_metrics['most_volatile_zone'])}"
        )

    rf_accuracy_text = 'Nicht verfuegbar'
    if prediction_artifacts.accuracy_mean is not None:
        rf_accuracy_text = f"{prediction_artifacts.accuracy_mean:.1%} +/- {prediction_artifacts.accuracy_std:.1%}"

    content = f"""# Qualitaetsreport

## Datensatz

- Datei: `{meta.file_path.name}`
- Zeilen: {meta.n_rows}
- Spalten: {meta.n_columns}
- Basis-Signale: {meta.base_signal_count}
- Zeitraum: {meta.start_time} bis {meta.end_time}
- Takt: {meta.step_seconds or 'unbekannt'} s

## Tagesueberblick

- Erkannte Coils: {summary['n_coils']}
- Schlechte Coils: {summary['n_bad_coils']}
- Auffaellige Coils: {summary.get('n_alert_coils', 0)}
- Produktionszyklen: {summary['n_cycles']}
- RF Accuracy: {rf_accuracy_text}

## Dickenabweichung nach Phase

{chr(10).join(thickness_lines)}

## Qualitaet nach Stich

{chr(10).join(stitch_lines)}

## Planheit

{planarity_section}

## Robuste Ausreisser

{chr(10).join(anomaly_lines)}

## Top Random-Forest-Features

{chr(10).join(top_rf_lines)}

## Top Korrelationen

{chr(10).join(top_corr_lines)}

## Was man besser machen kann

{chr(10).join(recommendation_lines)}

## Modellparameter

```json
{json.dumps(prediction_artifacts.rf_params, indent=2)}
```
"""
    report_path.write_text(content, encoding='utf-8')



def write_management_html_report(
    report_path: Path,
    meta,
    summary: dict[str, object],
    phase_summary: pd.DataFrame,
    cycle_summary: pd.DataFrame,
    thickness_summary: pd.DataFrame,
    anomaly_summary: pd.DataFrame,
    planarity_profile: pd.DataFrame,
    planarity_metrics: dict[str, float] | None,
    prediction_artifacts,
    recommendations: list[str],
) -> None:
    n_coils = summary.get('n_coils', 0) or 0
    n_bad_coils = summary.get('n_bad_coils', 0) or 0
    n_alert_coils = summary.get('n_alert_coils', 0) or 0
    bad_rate_pct = (n_bad_coils / n_coils * 100.0) if n_coils else 0.0
    production_hours = float(cycle_summary['duration_min'].sum() / 60.0) if not cycle_summary.empty else 0.0
    production_share = 0.0
    if not phase_summary.empty and 'Produktion' in phase_summary['phase_name'].values:
        production_share = float(phase_summary.loc[phase_summary['phase_name'] == 'Produktion', 'share_pct'].iloc[0])

    quality_by_stitch = prediction_artifacts.quality_by_stitch.copy() if prediction_artifacts.quality_by_stitch is not None else pd.DataFrame()
    worst_stitch_text = 'Keine Stich-Auswertung verfuegbar.'
    if not quality_by_stitch.empty:
        worst_stitch = quality_by_stitch.sort_values('bad_rate_pct', ascending=False).iloc[0]
        worst_stitch_text = (
            f"Stich {int(worst_stitch['stitch'])} zeigt die hoechste Fehlerquote mit {worst_stitch['bad_rate_pct']:.1f}%."
        )

    top_rf = prediction_artifacts.top_rf_features.copy() if prediction_artifacts.top_rf_features is not None else pd.DataFrame()
    if not top_rf.empty:
        top_rf['Treiber'] = top_rf['feature'].map(_pretty_feature_name)
        top_rf = top_rf[['Treiber', 'importance']].rename(columns={'importance': 'Importance'})

    top_corr = prediction_artifacts.top_correlations.copy() if prediction_artifacts.top_correlations is not None else pd.DataFrame()
    if not top_corr.empty:
        top_corr['Feature'] = top_corr['feature'].map(_pretty_feature_name)
        top_corr = top_corr[['Feature', 'correlation', 'abs_correlation']].rename(
            columns={'correlation': 'Korrelation', 'abs_correlation': '|Korrelation|'}
        )

    worst_coils = pd.DataFrame()
    feature_matrix = prediction_artifacts.feature_matrix
    if feature_matrix is not None and not feature_matrix.empty and 'thickness_sigma_pct' in feature_matrix.columns:
        sorted_coils = feature_matrix.dropna(subset=['thickness_sigma_pct']).sort_values(
            ['quality_rank', 'quality_score', 'local_peak_sigma_pct', 'share_abs_gt_2pct', 'max_abs_thickness_pct', 'thickness_sigma_pct'],
            ascending=[False, False, False, False, False, False],
        )
        display_columns = [
            column
            for column in [
                'coil_id',
                'stitch',
                'duration_min',
                'quality_score',
                'quality_flags',
                'thickness_sigma_pct',
                'local_peak_sigma_pct',
                'local_peak_share_abs_gt_2pct',
                'local_edge_center_shift',
                'quality_driver',
                'quality',
                'quality_rule',
            ]
            if column in sorted_coils.columns
        ]
        worst_coils = sorted_coils.head(10)[display_columns].rename(
            columns={
                'coil_id': 'Coil',
                'stitch': 'Stich',
                'duration_min': 'Dauer min',
                'quality_score': 'Score',
                'quality_flags': 'Flags',
                'thickness_sigma_pct': 'Sigma %',
                'local_peak_sigma_pct': 'Hotspot Sigma %',
                'local_peak_share_abs_gt_2pct': 'Hotspot Anteil |Abw.|>2% %',
                'local_edge_center_shift': 'Rand-Mitte Delta',
                'quality_driver': 'Treiber',
                'quality': 'Qualitaet',
                'quality_rule': 'Kurzregel',
            }
        )

    anomaly_table = anomaly_summary.rename(
        columns={'signal': 'Signal', 'anomaly_count': 'Ausreisser', 'anomaly_share_pct': 'Anteil %'}
    )
    stitch_table = quality_by_stitch.rename(
        columns={'stitch': 'Stich', 'total_coils': 'Coils gesamt', 'bad_coils': 'Schlecht', 'bad_rate_pct': 'Fehlerquote %'}
    )

    operator_focus = _build_operator_focus(recommendations, prediction_artifacts, anomaly_summary)
    management_figures = _build_management_figures(
        phase_summary=phase_summary,
        cycle_summary=cycle_summary,
        thickness_summary=thickness_summary,
        anomaly_summary=anomaly_summary,
        planarity_profile=planarity_profile,
        prediction_artifacts=prediction_artifacts,
    )

    rf_accuracy_text = 'Nicht verfuegbar'
    if prediction_artifacts.accuracy_mean is not None:
        rf_accuracy_text = f"{prediction_artifacts.accuracy_mean:.1%} +/- {prediction_artifacts.accuracy_std:.1%}"

    planarity_summary_html = '<p class="empty-state">Keine Planheitsdaten verfuegbar.</p>'
    if planarity_metrics is not None:
        planarity_summary_html = f"""
        <ul class="bullet-list">
          <li>Mitte: {planarity_metrics['center_mean']:.3f}</li>
          <li>Rand: {planarity_metrics['edge_mean']:.3f}</li>
          <li>Rand minus Mitte: {planarity_metrics['edge_minus_center']:.3f}</li>
          <li>Volatilste Zone: {int(planarity_metrics['most_volatile_zone'])}</li>
        </ul>
        """

    html_content = f"""<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gesamtreport Produktion und Qualitaet</title>
  <style>
    :root {{
      --bg: #efe9dc;
      --panel: #fffdf8;
      --ink: #1f1f1f;
      --muted: #645f57;
      --line: #d8cfbe;
      --accent: #b6462f;
      --accent-dark: #7c2d1f;
      --ok: #1c7c54;
      --warn: #bc6c25;
      --shadow: 0 18px 40px rgba(78, 58, 32, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: "Segoe UI", Arial, sans-serif; line-height: 1.45; }}
    .page {{ max-width: 1500px; margin: 0 auto; padding: 28px; }}
    .hero {{ background: linear-gradient(135deg, #fff8ee 0%, #f5e4cc 52%, #e5b68c 100%); border: 1px solid var(--line); border-radius: 24px; padding: 28px; box-shadow: var(--shadow); }}
    .hero h1 {{ margin: 0 0 10px; font-size: 38px; }}
    .hero p {{ margin: 6px 0; color: var(--muted); font-size: 16px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 16px; margin-top: 22px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 18px; box-shadow: var(--shadow); }}
    .card .label {{ color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .card .value {{ margin-top: 8px; font-size: 30px; font-weight: 700; }}
    .section {{ margin-top: 24px; background: var(--panel); border: 1px solid var(--line); border-radius: 24px; padding: 24px; box-shadow: var(--shadow); }}
    .section h2 {{ margin: 0 0 12px; font-size: 26px; }}
    .section h3 {{ margin: 0 0 10px; font-size: 20px; }}
    .section p {{ margin: 0 0 12px; color: var(--muted); }}
    .grid-two {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }}
    .grid-three {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }}
    .bullet-list {{ margin: 0; padding-left: 20px; }}
    .bullet-list li {{ margin: 8px 0; }}
    .callout {{ border-left: 6px solid var(--accent); padding-left: 18px; }}
    .plot-stack > div {{ margin-top: 16px; }}
    .data-table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    .data-table th, .data-table td {{ border-bottom: 1px solid var(--line); padding: 9px 10px; text-align: left; vertical-align: top; }}
    .data-table th {{ background: #f5efe3; position: sticky; top: 0; }}
    .empty-state {{ color: var(--muted); font-style: italic; }}
    .badge-ok {{ color: var(--ok); font-weight: 700; }}
    .badge-warn {{ color: var(--warn); font-weight: 700; }}
    .footer-note {{ margin-top: 18px; font-size: 13px; color: var(--muted); }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>Gesamtreport Produktion und Qualitaet</h1>
      <p>Datei: <strong>{html.escape(meta.file_path.name)}</strong></p>
      <p>Zeitraum: <strong>{html.escape(str(meta.start_time))}</strong> bis <strong>{html.escape(str(meta.end_time))}</strong></p>
      <p>Dieser Report ist fuer Produktionsleitung und Maschinenfahrer gedacht und fasst die wichtigsten Risiken, Treiber und Handlungspunkte auf einer Seite zusammen.</p>
      <div class="cards">
        <div class="card"><div class="label">Coils gesamt</div><div class="value">{n_coils}</div></div>
        <div class="card"><div class="label">Schlechte Coils</div><div class="value">{n_bad_coils}</div></div>
        <div class="card"><div class="label">Auffaellige Coils</div><div class="value">{n_alert_coils}</div></div>
        <div class="card"><div class="label">Fehlerquote</div><div class="value">{bad_rate_pct:.1f}%</div></div>
        <div class="card"><div class="label">Produktionszyklen</div><div class="value">{summary.get('n_cycles', 0)}</div></div>
        <div class="card"><div class="label">Produktionszeit</div><div class="value">{production_hours:.1f} h</div></div>
        <div class="card"><div class="label">Anteil Produktion</div><div class="value">{production_share:.1f}%</div></div>
        <div class="card"><div class="label">RF Accuracy</div><div class="value">{html.escape(rf_accuracy_text)}</div></div>
      </div>
    </section>

    <section class="section">
      <h2>Management Summary</h2>
      <div class="callout">
        <p>{html.escape(worst_stitch_text)}</p>
        <p>Die aktuelle Auswertung zeigt eine deutlich hoehere Dickenstreuung im Anlauf als in der stabilen Produktion. Gleichzeitig liegen mehrere Top-Praediktoren im Anlauf-Fenster und im Bereich der rechten Shapemeter-Randzonen.</p>
        <p>Der Report unten zeigt deshalb nicht nur die Qualitaetslage am Ende, sondern auch die Signale, die frueh auf Probleme hinweisen.</p>
      </div>
    </section>

    <section class="section grid-two">
      <div>
        <h2>Wichtigste Massnahmen</h2>
        <ul class="bullet-list">
          {''.join(f'<li>{html.escape(item)}</li>' for item in recommendations)}
        </ul>
      </div>
      <div>
        <h2>Fokus Fuer Maschinenfahrer</h2>
        <ul class="bullet-list">
          {''.join(f'<li>{html.escape(item)}</li>' for item in operator_focus)}
        </ul>
      </div>
    </section>

    <section class="section">
      <h2>Visual Overview</h2>
      <p>Die folgenden Diagramme bilden die wichtigsten Prozess-, Qualitaets- und Modellinformationen in einer langen Gesamtansicht ab.</p>
      <div class="plot-stack">
        {''.join(management_figures)}
      </div>
    </section>

    <section class="section grid-three">
      <div>
        <h3>Qualitaet Nach Stich</h3>
        {_table_html(stitch_table, max_rows=10)}
      </div>
      <div>
        <h3>Auffaelligste Coils Nach Qualitaetsregeln</h3>
        {_table_html(worst_coils, max_rows=10)}
      </div>
      <div>
        <h3>Top Ausreisser</h3>
        {_table_html(anomaly_table, max_rows=10)}
      </div>
    </section>

    <section class="section grid-two">
      <div>
        <h2>Top Random-Forest-Treiber</h2>
        <p>Diese Features erklaeren am staerksten, warum ein Coil spaeter als gut oder schlecht eingeordnet wird.</p>
        {_table_html(top_rf, max_rows=12)}
      </div>
      <div>
        <h2>Top Korrelationen</h2>
        <p>Diese Tabelle zeigt die staerksten linearen Zusammenhaenge mit dem Qualitaetslabel.</p>
        {_table_html(top_corr, max_rows=12)}
      </div>
    </section>

    <section class="section grid-two">
      <div>
        <h2>Planheit</h2>
        <p>Besonders relevant fuer Fuehrung und Fahrer ist der Vergleich von Bandmitte und Randzonen.</p>
        {planarity_summary_html}
      </div>
      <div>
        <h2>Technische Rahmendaten</h2>
        <ul class="bullet-list">
          <li>Zeilen: {meta.n_rows}</li>
          <li>Spalten: {meta.n_columns}</li>
          <li>Basis-Signale: {meta.base_signal_count}</li>
          <li>Takt: {meta.step_seconds or '-'} s</li>
          <li>Input-Datei: {html.escape(str(meta.file_path))}</li>
          <li>Modellparameter: n_estimators={prediction_artifacts.rf_params.get('n_estimators')}, max_depth={prediction_artifacts.rf_params.get('max_depth')}</li>
        </ul>
      </div>
    </section>

    <section class="section">
      <h2>Hinweis Zur Interpretation</h2>
      <p>Dieser Report zeigt datenbasierte Hinweise, keine kausal bewiesenen Zusammenhaenge. Besonders die Prediction-Abschnitte sind als Fruehwarnsystem zu lesen: Sie zeigen, welche Signale frueh mit schlechter Qualitaet zusammenlaufen, nicht automatisch die alleinige Ursache.</p>
      <p class="footer-note">Generiert aus derselben Pipeline wie Report, Prediction und Einzelplots. Damit haben Produktionsleitung und Maschinefuehrer eine gemeinsame Sicht auf denselben Datensatz.</p>
    </section>
  </div>
</body>
</html>
"""
    report_path.write_text(html_content, encoding='utf-8')



