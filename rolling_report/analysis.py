from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .config import MIN_STATE_SECONDS, PHASE_SPEED_THRESHOLD


def sanitize_feature_name(name: str) -> str:
    clean = (
        name.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue')
        .replace('Ä', 'Ae').replace('Ö', 'Oe').replace('Ü', 'Ue').replace('ß', 'ss')
    )
    clean = clean.replace(' ', '_').replace('.', '_').replace('/', '_').replace('-', '_')
    clean = clean.replace('%', 'pct').replace('[', '_').replace(']', '_')
    while '__' in clean:
        clean = clean.replace('__', '_')
    return clean.strip('_').lower()



def find_exact_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    column_list = list(columns)
    for candidate in candidates:
        if candidate in column_list:
            return candidate
    return None



def get_shapemeter_columns(df: pd.DataFrame) -> list[str]:
    columns = [
        column
        for column in df.columns
        if 'SFC__gCurves_delTen[' in column and not column.endswith('.max') and not column.endswith('.min')
    ]
    return sorted(columns, key=lambda value: int(value.split('[')[-1].split(']')[0]))



def add_phase_information(
    df: pd.DataFrame,
    speed_column: str = 'Walzgeschwindigkeit akt.',
    speed_threshold: float = PHASE_SPEED_THRESHOLD,
    min_state_seconds: int = MIN_STATE_SECONDS,
    step_seconds: float | None = 10.0,
) -> pd.DataFrame:
    result = df.copy()
    if speed_column not in result.columns:
        result['phase_name'] = 'Unbekannt'
        return result

    speeds = pd.to_numeric(result[speed_column], errors='coerce').fillna(0.0).to_numpy()
    interval = step_seconds or 10.0
    min_steps = max(1, int(round(min_state_seconds / interval)))

    initial_state = 'Produktion' if len(speeds) > 0 and speeds[0] >= speed_threshold else 'Stillstand'
    phases: list[str] = [initial_state] * len(result)
    state = initial_state
    state_start = 0

    for idx, speed in enumerate(speeds):
        if state == 'Stillstand':
            phases[idx] = 'Stillstand'
            if speed >= speed_threshold:
                state = 'Anlauf'
                state_start = idx
        elif state == 'Anlauf':
            phases[idx] = 'Anlauf'
            if speed < speed_threshold:
                for back_idx in range(state_start, idx + 1):
                    phases[back_idx] = 'Stillstand'
                state = 'Stillstand'
                state_start = idx
            elif idx - state_start >= min_steps:
                state = 'Produktion'
                state_start = idx
        elif state == 'Produktion':
            phases[idx] = 'Produktion'
            if speed < speed_threshold:
                state = 'Auslauf'
                state_start = idx
        elif state == 'Auslauf':
            phases[idx] = 'Auslauf'
            if speed >= speed_threshold:
                for back_idx in range(state_start, idx + 1):
                    phases[back_idx] = 'Produktion'
                state = 'Produktion'
                state_start = idx
            elif idx - state_start >= min_steps:
                state = 'Stillstand'
                state_start = idx

    result['phase_name'] = phases
    return result



def summarize_phases(df: pd.DataFrame, step_seconds: float | None) -> pd.DataFrame:
    step = step_seconds or 10.0
    summary = (
        df['phase_name']
        .value_counts(dropna=False)
        .rename_axis('phase_name')
        .reset_index(name='samples')
        .sort_values('phase_name')
    )
    summary['duration_min'] = summary['samples'] * step / 60.0
    summary['share_pct'] = summary['samples'] / len(df) * 100.0
    return summary



def detect_cycles(df: pd.DataFrame, time_column: str, step_seconds: float | None) -> pd.DataFrame:
    interval = step_seconds or 10.0
    production_mask = df['phase_name'] == 'Produktion'
    if not production_mask.any():
        return pd.DataFrame(columns=['cycle_id', 'start_time', 'end_time', 'duration_min'])

    groups = (production_mask != production_mask.shift(fill_value=False)).cumsum()
    rows: list[dict[str, object]] = []
    cycle_id = 1
    for _, segment in df.loc[production_mask].groupby(groups[production_mask], sort=True):
        rows.append(
            {
                'cycle_id': cycle_id,
                'start_time': segment[time_column].iloc[0],
                'end_time': segment[time_column].iloc[-1],
                'duration_min': len(segment) * interval / 60.0,
            }
        )
        cycle_id += 1
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result[result['duration_min'] >= 5.0].reset_index(drop=True)
    result['cycle_id'] = range(1, len(result) + 1)
    return result



def _first_non_zero_value(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors='coerce')
    numeric = numeric[(~numeric.isna()) & (numeric != 0)]
    if numeric.empty:
        return None
    return float(numeric.iloc[0])



def detect_coils(df: pd.DataFrame, time_column: str, step_seconds: float | None) -> pd.DataFrame:
    if 'Bundanfang' not in df.columns:
        return pd.DataFrame(
            columns=[
                'coil_id',
                'start_idx',
                'end_idx',
                'start_time',
                'end_time',
                'duration_min',
                'stitch',
                'einlaufdicke_um',
                'solldicke_um',
                'bandbreite_mm',
            ]
        )

    interval = step_seconds or 10.0
    signal = pd.to_numeric(df['Bundanfang'], errors='coerce').fillna(0.0)
    edges = signal.gt(0.5) & signal.shift(fill_value=float(signal.iloc[0])).le(0.5)
    start_indices = list(df.index[edges])
    if not start_indices:
        return pd.DataFrame()

    stitch_column = find_exact_column(df.columns, ['Stich_Nr', 'Stichnummer'])
    rows: list[dict[str, object]] = []
    for position, start_idx in enumerate(start_indices, start=1):
        end_idx = start_indices[position] - 1 if position < len(start_indices) else int(df.index[-1])
        segment = df.iloc[start_idx : end_idx + 1]
        rows.append(
            {
                'coil_id': position,
                'start_idx': int(start_idx),
                'end_idx': int(end_idx),
                'start_time': segment[time_column].iloc[0],
                'end_time': segment[time_column].iloc[-1],
                'duration_min': len(segment) * interval / 60.0,
                'stitch': int(stitch_value) if ((stitch_value := _first_non_zero_value(segment[stitch_column])) is not None) else None,
                'einlaufdicke_um': _first_non_zero_value(segment['Einlaufdicke']) if 'Einlaufdicke' in df.columns else None,
                'solldicke_um': _first_non_zero_value(segment['Solldicke']) if 'Solldicke' in df.columns else None,
                'bandbreite_mm': _first_non_zero_value(segment['Bandbreite']) if 'Bandbreite' in df.columns else None,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result[result['duration_min'] >= 5.0].reset_index(drop=True)
    result['cycle_id'] = range(1, len(result) + 1)
    return result



def build_thickness_summary(df: pd.DataFrame) -> pd.DataFrame:
    thickness_column = find_exact_column(df.columns, ['Dickenabweichung'])
    if thickness_column is None or 'phase_name' not in df.columns:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for phase_name, segment in df.groupby('phase_name', sort=False):
        values = pd.to_numeric(segment[thickness_column], errors='coerce').dropna()
        if values.empty:
            continue
        rows.append(
            {
                'phase_name': phase_name,
                'samples': int(values.shape[0]),
                'mean_pct': float(values.mean()),
                'std_pct': float(values.std(ddof=0)),
                'min_pct': float(values.min()),
                'max_pct': float(values.max()),
                'count_abs_gt_1pct': int(values.abs().gt(1.0).sum()),
                'count_abs_gt_2pct': int(values.abs().gt(2.0).sum()),
            }
        )
    return pd.DataFrame(rows)
def build_planarity_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float] | None]:
    zone_columns = get_shapemeter_columns(df)
    if not zone_columns:
        return pd.DataFrame(), None

    numeric = df[zone_columns].apply(pd.to_numeric, errors='coerce')
    profile = pd.DataFrame(
        {
            'zone': [int(column.split('[')[-1].split(']')[0]) for column in zone_columns],
            'mean_value': numeric.mean().to_list(),
            'std_value': numeric.std(ddof=0).to_list(),
        }
    )
    center_slice = profile.loc[profile['zone'].between(13, 17), 'mean_value']
    edge_slice = pd.concat(
        [profile.loc[profile['zone'].between(1, 3), 'mean_value'], profile.loc[profile['zone'].between(27, 29), 'mean_value']]
    )
    metrics = {
        'center_mean': float(center_slice.mean()),
        'edge_mean': float(edge_slice.mean()),
        'edge_minus_center': float(edge_slice.mean() - center_slice.mean()),
        'most_volatile_zone': float(profile.sort_values('std_value', ascending=False)['zone'].iloc[0]),
    }
    return profile, metrics



def build_anomaly_summary(df: pd.DataFrame) -> pd.DataFrame:
    candidate_columns = [
        'Walzkraft AS akt.',
        'Walzkraft BS akt.',
        'Druck AS',
        'Druck BS',
        'Spannung DMD AS',
        'Spannung DMD BS',
        'Biegung akt.',
    ]
    source = df.loc[df['phase_name'] == 'Produktion'].copy() if 'phase_name' in df.columns else df
    rows: list[dict[str, object]] = []
    for column in candidate_columns:
        if column not in source.columns:
            continue
        values = pd.to_numeric(source[column], errors='coerce').dropna()
        if values.empty:
            continue
        median = values.median()
        mad = (values - median).abs().median()
        if mad and mad > 0:
            score = 0.6745 * (values - median) / mad
        else:
            std = values.std(ddof=0)
            if std == 0 or pd.isna(std):
                score = pd.Series(np.zeros(len(values)), index=values.index)
            else:
                score = (values - values.mean()) / std
        rows.append(
            {
                'signal': column,
                'anomaly_count': int(score.abs().gt(3.5).sum()),
                'anomaly_share_pct': float(score.abs().gt(3.5).mean() * 100.0),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values('anomaly_count', ascending=False)



def build_recommendations(
    thickness_summary: pd.DataFrame,
    prediction_summary: dict[str, object],
    planarity_metrics: dict[str, float] | None,
    anomaly_summary: pd.DataFrame,
) -> list[str]:
    recommendations: list[str] = []

    if not thickness_summary.empty:
        startup_std = thickness_summary.loc[thickness_summary['phase_name'] == 'Anlauf', 'std_pct']
        production_std = thickness_summary.loc[thickness_summary['phase_name'] == 'Produktion', 'std_pct']
        if not startup_std.empty and not production_std.empty and startup_std.iloc[0] > production_std.iloc[0] * 1.15:
            recommendations.append(
                'Anlaufphase stabilisieren: Die Dickenstreuung im Anlauf ist deutlich hoeher als in der Produktion.'
            )

    stitch_table = prediction_summary.get('quality_by_stitch')
    if isinstance(stitch_table, pd.DataFrame) and not stitch_table.empty:
        worst = stitch_table.sort_values('bad_rate_pct', ascending=False).iloc[0]
        if worst['bad_rate_pct'] > 0:
            recommendations.append(
                f"Stich {int(worst['stitch'])} priorisieren: Dort liegt die hoechste Fehlerquote mit {worst['bad_rate_pct']:.1f}%."
            )

    top_features = prediction_summary.get('top_rf_features')
    if isinstance(top_features, pd.DataFrame) and not top_features.empty:
        feature_names = top_features['feature'].astype(str)
        startup_hits = feature_names.str.contains('startup').sum()
        if startup_hits >= 2:
            recommendations.append(
                'Die ersten 60 Sekunden aktiv ueberwachen: Mehrere Top-Praediktoren kommen aus dem Anlauf-Fenster.'
            )
        if feature_names.str.contains('local_|_shift').any():
            recommendations.append(
                'Lokale Stoerfenster je Coil getrennt auswerten: Kurze Qualitaetseinbrueche koennen im Coil-Mittel untergehen.'
            )
        if feature_names.str.contains('edge_center|left_right|gcurves_delten_27|gcurves_delten_28').any():
            recommendations.append(
                'Rand-Mitte-Planheit im Stoerfenster enger beobachten: Mehrere Treiber kommen aus den Shapemeter-Randzonen.'
            )

    if planarity_metrics is not None and abs(planarity_metrics['edge_minus_center']) > 0.5:
        recommendations.append(
            'Bandrand-Planheit genauer beobachten: Die Randzonen weichen spuerbar von der Bandmitte ab.'
        )

    if not anomaly_summary.empty:
        top_signal = anomaly_summary.sort_values('anomaly_count', ascending=False).iloc[0]
        if top_signal['anomaly_count'] > 0:
            recommendations.append(
                f"Signal '{top_signal['signal']}' technisch pruefen: Es zeigt die meisten robusten Ausreisser."
            )

    if not recommendations:
        recommendations.append('Keine dominante Einzelursache gefunden. Weitere Tage einbeziehen, um robuste Trends abzusichern.')

    return recommendations






