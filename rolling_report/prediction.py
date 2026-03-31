from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .analysis import get_shapemeter_columns, sanitize_feature_name
from .config import (
    DEFAULT_QUALITY_THRESHOLD,
    QUALITY_LOCAL_EDGE_CENTER_SHIFT_THRESHOLD,
    QUALITY_LOCAL_MIN_WINDOW_SAMPLES,
    QUALITY_LOCAL_PEAK_SHARE_ABS_GT_2PCT_THRESHOLD,
    QUALITY_LOCAL_PEAK_SIGMA_THRESHOLD,
    QUALITY_LOCAL_WINDOW_SECONDS,
    QUALITY_MIN_EVIDENCE_FOR_BAD,
    QUALITY_SHARE_ABS_GT_2PCT_THRESHOLD,
    QUALITY_SHARE_BAD_THRESHOLD,
    QUALITY_THRESHOLDS,
    RANDOM_FOREST_DEFAULTS,
)


PROCESS_SIGNAL_CANDIDATES = [
    ['Walzkraft AS akt.'],
    ['Walzkraft BS akt.'],
    ['Summenwalzkraft akt.'],
    ['Differenzkraft'],
    ['Druck AS'],
    ['Druck BS'],
    ['Spannung DMD AS'],
    ['Spannung DMD BS'],
    ['Walzgeschwindigkeit akt.'],
    ['ref. Walzgeschwindigkeit'],
    ['Abhaspelzug akt.'],
    ['Aufhaspelzug akt.'],
    ['dvdt'],
    ['Biegung akt.'],
    ['Biegeregler Output'],
    ['Schraeglageregler Output', 'SchrÃ¤glageregler Output'],
    ['Differenzposition'],
    ['Differenzposition AS'],
    ['Differenzposition BS'],
    ['Walzoeldruck', 'Walzoldruck'],
    ['Temperatur Walzoel', 'Temperatur Walzol'],
    ['Strom Hauptantrieb'],
    ['Leistung Hauptantrieb'],
]

LEAKAGE_PREFIXES = (
    'dickenabweichung',
    'abs_dickenabweichung',
    'absolutdicke',
    'ims_istdicke_mk1',
    'ims_aktuelle_mittendicke',
)


@dataclass
class PredictionArtifacts:
    feature_matrix: pd.DataFrame
    quality_by_stitch: pd.DataFrame
    top_correlations: pd.DataFrame
    top_rf_features: pd.DataFrame
    accuracy_mean: float | None
    accuracy_std: float | None
    rf_params: dict[str, object]



def _resolve_column_name(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {sanitize_feature_name(str(column)): str(column) for column in columns}
    for candidate in candidates:
        match = normalized.get(sanitize_feature_name(str(candidate)))
        if match is not None:
            return match
    return None



def _numeric_series(frame: pd.DataFrame, candidates: str | Iterable[str]) -> pd.Series:
    candidate_list = [candidates] if isinstance(candidates, str) else list(candidates)
    column = _resolve_column_name(frame.columns, candidate_list)
    if column is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors='coerce').dropna()



def _safe_std(series: pd.Series) -> float | None:
    if series.empty:
        return None
    value = float(series.std(ddof=0))
    return value if not np.isnan(value) else 0.0



def _safe_mean(series: pd.Series) -> float | None:
    if series.empty:
        return None
    value = float(series.mean())
    return value if not np.isnan(value) else None



def _safe_mean_abs(series: pd.Series) -> float | None:
    if series.empty:
        return None
    value = float(series.abs().mean())
    return value if not np.isnan(value) else None



def _safe_quantile_abs(series: pd.Series, quantile: float) -> float | None:
    if series.empty:
        return None
    value = float(series.abs().quantile(quantile))
    return value if not np.isnan(value) else None



def _safe_max_abs(series: pd.Series) -> float | None:
    if series.empty:
        return None
    value = float(series.abs().max())
    return value if not np.isnan(value) else None



def _share_abs_gt(series: pd.Series, threshold: float) -> float | None:
    if series.empty:
        return None
    return float(series.abs().gt(threshold).mean() * 100.0)



def _window_rows(interval: float) -> int:
    return max(QUALITY_LOCAL_MIN_WINDOW_SAMPLES, int(round(QUALITY_LOCAL_WINDOW_SECONDS / interval)))



def _quality_window_index(thickness_values: pd.Series, interval: float) -> pd.Index:
    if thickness_values.empty:
        return thickness_values.index

    window_rows = _window_rows(interval)
    if len(thickness_values) < window_rows:
        return thickness_values.index

    rolling_sigma = thickness_values.rolling(window_rows, min_periods=window_rows).std(ddof=0)
    if not rolling_sigma.notna().any():
        return thickness_values.index

    peak_end_position = int(np.nanargmax(rolling_sigma.to_numpy(dtype=float)))
    start_position = max(0, peak_end_position - window_rows + 1)
    return thickness_values.index[start_position : peak_end_position + 1]



def _shape_zone_columns(shape_columns: list[str]) -> tuple[list[str], list[str], list[str]]:
    zone_pairs: list[tuple[str, int | None]] = []
    for column in shape_columns:
        try:
            zone = int(column.split('[')[-1].split(']')[0])
        except ValueError:
            zone = None
        zone_pairs.append((column, zone))

    left = [column for column, zone in zone_pairs if zone is not None and zone <= 3]
    center = [column for column, zone in zone_pairs if zone is not None and 13 <= zone <= 17]
    right = [column for column, zone in zone_pairs if zone is not None and zone >= 27]
    if left and center and right:
        return left, center, right

    third = max(1, len(shape_columns) // 3)
    left = shape_columns[:third]
    right = shape_columns[len(shape_columns) - third :]
    center = shape_columns[third : len(shape_columns) - third]
    if not center:
        center = shape_columns[third : third + max(1, len(shape_columns) - 2 * third)]
    return left, center, right



def _shape_mean(frame: pd.DataFrame, columns: list[str]) -> float | None:
    if not columns or frame.empty:
        return None
    numeric = frame[columns].apply(pd.to_numeric, errors='coerce')
    if numeric.empty:
        return None
    value = float(numeric.mean().mean())
    return value if not np.isnan(value) else None



def _shape_profile_std(frame: pd.DataFrame, columns: list[str]) -> float | None:
    if not columns or frame.empty:
        return None
    numeric = frame[columns].apply(pd.to_numeric, errors='coerce')
    if numeric.empty:
        return None
    value = float(numeric.std(ddof=0).mean())
    return value if not np.isnan(value) else None



def _segment_process_metrics(frame: pd.DataFrame, shape_columns: list[str]) -> dict[str, float | None]:
    sum_force = _safe_mean(_numeric_series(frame, ['Summenwalzkraft akt.']))
    diff_force = _safe_mean(_numeric_series(frame, ['Differenzkraft']))
    pressure_as = _safe_mean(_numeric_series(frame, ['Druck AS']))
    pressure_bs = _safe_mean(_numeric_series(frame, ['Druck BS']))
    tension_as = _safe_mean(_numeric_series(frame, ['Spannung DMD AS']))
    tension_bs = _safe_mean(_numeric_series(frame, ['Spannung DMD BS']))
    speed_actual = _safe_mean(_numeric_series(frame, ['Walzgeschwindigkeit akt.']))
    speed_reference = _safe_mean(_numeric_series(frame, ['ref. Walzgeschwindigkeit']))
    bending_activity = _safe_std(_numeric_series(frame, ['Biegeregler Output']))
    slant_activity = _safe_std(_numeric_series(frame, ['Schraeglageregler Output', 'SchrÃ¤glageregler Output']))
    dvdt_activity = _safe_std(_numeric_series(frame, ['dvdt']))

    force_imbalance = None
    if sum_force is not None and sum_force != 0 and diff_force is not None:
        force_imbalance = abs(diff_force) / abs(sum_force) * 100.0

    pressure_gap = None
    if pressure_as is not None and pressure_bs is not None:
        pressure_gap = abs(pressure_as - pressure_bs)

    tension_gap = None
    if tension_as is not None and tension_bs is not None:
        tension_gap = abs(tension_as - tension_bs)

    speed_tracking_error = None
    if speed_actual is not None and speed_reference is not None:
        speed_tracking_error = abs(speed_actual - speed_reference)

    edge_center = None
    left_right = None
    shape_std = None
    if shape_columns:
        left_columns, center_columns, right_columns = _shape_zone_columns(shape_columns)
        edge_mean = _shape_mean(frame, left_columns + right_columns)
        center_mean = _shape_mean(frame, center_columns)
        left_mean = _shape_mean(frame, left_columns)
        right_mean = _shape_mean(frame, right_columns)
        if edge_mean is not None and center_mean is not None:
            edge_center = edge_mean - center_mean
        if left_mean is not None and right_mean is not None:
            left_right = abs(left_mean - right_mean)
        shape_std = _shape_profile_std(frame, shape_columns)

    return {
        'force_imbalance_pct': force_imbalance,
        'pressure_gap': pressure_gap,
        'tension_gap': tension_gap,
        'speed_tracking_error': speed_tracking_error,
        'bending_activity': bending_activity,
        'slant_activity': slant_activity,
        'dvdt_activity': dvdt_activity,
        'edge_center': edge_center,
        'left_right': left_right,
        'shape_profile_std': shape_std,
    }



def _metric_shift(local_value: float | None, baseline_value: float | None) -> float | None:
    if local_value is None or baseline_value is None:
        return None
    return local_value - baseline_value



def _summarize_quality_drivers(
    local_metrics: dict[str, float | None],
    baseline_metrics: dict[str, float | None],
) -> tuple[str, str]:
    driver_specs = [
        ('edge_center', 'Rand-Mitte-Planheit', '', 0.30, True),
        ('left_right', 'Links-Rechts-Planheit', '', 0.20, False),
        ('shape_profile_std', 'Planheitsvolatilitaet', '', 0.10, False),
        ('bending_activity', 'Biegeregler-Aktivitaet', '', 0.15, False),
        ('slant_activity', 'Schraeglageregler-Aktivitaet', '', 0.15, False),
        ('pressure_gap', 'Druckdifferenz AS/BS', '', 0.10, False),
        ('tension_gap', 'Zugdifferenz AS/BS', '', 0.005, False),
        ('speed_tracking_error', 'Geschwindigkeitsfehler', '', 0.25, False),
        ('force_imbalance_pct', 'Differenzkraftanteil', '%', 0.05, False),
        ('dvdt_activity', 'dVdt-Aktivitaet', '', 0.05, False),
    ]

    details: list[tuple[float, str, float, str]] = []
    for key, label, unit, tolerance, compare_abs in driver_specs:
        local_value = local_metrics.get(key)
        baseline_value = baseline_metrics.get(key)
        if local_value is None or baseline_value is None:
            continue
        if compare_abs:
            delta = abs(local_value) - abs(baseline_value)
        else:
            delta = local_value - baseline_value
        if delta < tolerance:
            continue
        details.append((delta, label, delta, unit))

    if not details:
        return 'Keine dominante Prozessabweichung', 'Keine dominante Prozessabweichung im lokalen Stoerfenster'

    details.sort(key=lambda item: item[0], reverse=True)
    primary_driver = details[0][1]
    detail_text = '; '.join(f'{label} {delta:+.2f}{unit}' for _, label, delta, unit in details[:3])
    return primary_driver, detail_text



def _build_local_quality_metrics(
    segment: pd.DataFrame,
    interval: float,
    time_column: str,
    shape_columns: list[str],
) -> dict[str, object]:
    thickness_values = _numeric_series(segment, 'Dickenabweichung')
    if thickness_values.empty:
        return {
            'local_quality_window_seconds': None,
            'local_window_start_time': None,
            'local_window_end_time': None,
            'local_peak_sigma_pct': None,
            'local_peak_mean_abs_thickness_pct': None,
            'local_peak_q95_abs_thickness_pct': None,
            'local_peak_max_abs_thickness_pct': None,
            'local_peak_share_abs_gt_2pct': None,
            'quality_driver': 'Keine Dickenabweichungswerte',
            'quality_driver_detail': 'Keine Dickenabweichungswerte',
        }

    window_index = _quality_window_index(thickness_values, interval)
    local_segment = segment.loc[window_index]
    baseline_segment = segment.drop(index=window_index)
    if baseline_segment.empty:
        baseline_segment = segment

    local_thickness = _numeric_series(local_segment, 'Dickenabweichung')
    baseline_thickness = _numeric_series(baseline_segment, 'Dickenabweichung')
    local_metrics = _segment_process_metrics(local_segment, shape_columns)
    baseline_metrics = _segment_process_metrics(baseline_segment, shape_columns)
    quality_driver, quality_driver_detail = _summarize_quality_drivers(local_metrics, baseline_metrics)

    result: dict[str, object] = {
        'local_quality_window_seconds': float(len(local_segment) * interval),
        'local_window_start_time': local_segment[time_column].iloc[0] if time_column in local_segment.columns and not local_segment.empty else None,
        'local_window_end_time': local_segment[time_column].iloc[-1] if time_column in local_segment.columns and not local_segment.empty else None,
        'local_peak_sigma_pct': _safe_std(local_thickness),
        'local_peak_mean_abs_thickness_pct': _safe_mean_abs(local_thickness),
        'local_peak_q95_abs_thickness_pct': _safe_quantile_abs(local_thickness, 0.95),
        'local_peak_max_abs_thickness_pct': _safe_max_abs(local_thickness),
        'local_peak_share_abs_gt_2pct': _share_abs_gt(local_thickness, 2.0),
        'baseline_sigma_pct': _safe_std(baseline_thickness),
        'quality_driver': quality_driver,
        'quality_driver_detail': quality_driver_detail,
    }

    for key, value in local_metrics.items():
        result[f'local_{key}'] = value
    for key, value in baseline_metrics.items():
        result[f'baseline_{key}'] = value
    for key in local_metrics:
        result[f'local_{key}_shift'] = _metric_shift(local_metrics.get(key), baseline_metrics.get(key))

    return result



def _compute_coil_label(
    thickness_sigma: float | None,
    stitch: int | None,
    max_abs_thickness: float | None,
    share_abs_gt_2pct: float | None,
    local_peak_sigma: float | None,
    local_peak_share_abs_gt_2pct: float | None,
    local_edge_center_shift: float | None,
    local_window_seconds: float | None,
) -> tuple[str, int | None, int | None, int, float, int, str, str, str]:
    threshold = QUALITY_THRESHOLDS.get(stitch, DEFAULT_QUALITY_THRESHOLD)
    if (
        thickness_sigma is None
        and max_abs_thickness is None
        and share_abs_gt_2pct is None
        and local_peak_sigma is None
    ):
        return (
            'unbekannt',
            None,
            None,
            -1,
            threshold,
            0,
            'Keine Daten',
            'Keine Dickenabweichungswerte',
            'Keine Dickenabweichungswerte',
        )

    sigma_flag = thickness_sigma is not None and thickness_sigma > threshold
    share_warning_flag = share_abs_gt_2pct is not None and share_abs_gt_2pct > QUALITY_SHARE_ABS_GT_2PCT_THRESHOLD
    share_bad_flag = share_abs_gt_2pct is not None and share_abs_gt_2pct >= QUALITY_SHARE_BAD_THRESHOLD
    edge_flag = local_edge_center_shift is not None and abs(local_edge_center_shift) > QUALITY_LOCAL_EDGE_CENTER_SHIFT_THRESHOLD
    hotspot_share_flag = (
        local_peak_share_abs_gt_2pct is not None
        and local_peak_share_abs_gt_2pct >= QUALITY_LOCAL_PEAK_SHARE_ABS_GT_2PCT_THRESHOLD
    )
    hotspot_flag = (
        local_peak_sigma is not None
        and local_peak_sigma > QUALITY_LOCAL_PEAK_SIGMA_THRESHOLD
        and (edge_flag or hotspot_share_flag)
    )
    score = int(sigma_flag) + int(share_bad_flag) + int(hotspot_flag) + int(edge_flag)

    display_edge = edge_flag and (sigma_flag or share_warning_flag or hotspot_flag)
    flags: list[str] = []
    details: list[str] = []
    if sigma_flag:
        flags.append('Sigma')
        details.append(f'Sigma {thickness_sigma:.2f}% > {threshold:.2f}%')
    if share_bad_flag:
        flags.append('Share>=5.0')
        details.append(f'Anteil |Abw.|>2% {share_abs_gt_2pct:.2f}% >= {QUALITY_SHARE_BAD_THRESHOLD:.2f}%')
    elif share_warning_flag:
        flags.append('Share>4.5')
        details.append(f'Anteil |Abw.|>2% {share_abs_gt_2pct:.2f}% > {QUALITY_SHARE_ABS_GT_2PCT_THRESHOLD:.2f}%')
    if hotspot_flag:
        flags.append('Hotspot')
        hotspot_parts = [
            f'Lokales {int(local_window_seconds or QUALITY_LOCAL_WINDOW_SECONDS)}s-Sigma {local_peak_sigma:.2f}% > {QUALITY_LOCAL_PEAK_SIGMA_THRESHOLD:.2f}%'
        ]
        if hotspot_share_flag:
            hotspot_parts.append(
                f'lokaler Anteil |Abw.|>2% {local_peak_share_abs_gt_2pct:.2f}% >= {QUALITY_LOCAL_PEAK_SHARE_ABS_GT_2PCT_THRESHOLD:.2f}%'
            )
        if edge_flag:
            hotspot_parts.append(
                f'Rand-Mitte-Sprung {local_edge_center_shift:+.2f} > {QUALITY_LOCAL_EDGE_CENTER_SHIFT_THRESHOLD:.2f}'
            )
        details.append('; '.join(hotspot_parts))
    if display_edge:
        flags.append('Edge')

    if score >= QUALITY_MIN_EVIDENCE_FOR_BAD:
        quality_text = 'schlecht'
        quality_label = 1
        attention_label = 1
        quality_rank = 2
        quality_rule = f"Score {score}: {' + '.join(flags)}"
    elif sigma_flag or share_warning_flag or hotspot_flag:
        quality_text = 'auffaellig'
        quality_label = None
        attention_label = 1
        quality_rank = 1
        quality_rule = f"Warnung: {', '.join(flags)}"
    else:
        quality_text = 'gut'
        quality_label = 0
        attention_label = 0
        quality_rank = 0
        quality_rule = 'Innerhalb der Qualitaetsgrenzen'

    quality_flags = ' | '.join(flags) if flags else 'OK'
    quality_detail = '; '.join(details) if details else 'Keine starken Warnsignale'
    return (
        quality_text,
        quality_label,
        attention_label,
        quality_rank,
        threshold,
        score,
        quality_flags,
        quality_rule,
        quality_detail,
    )



def build_feature_matrix(
    df: pd.DataFrame,
    coils: pd.DataFrame,
    step_seconds: float | None,
    startup_seconds: int,
    time_column: str,
) -> pd.DataFrame:
    if coils.empty:
        return pd.DataFrame()

    interval = step_seconds or 10.0
    startup_rows = max(1, int(round(startup_seconds / interval)))
    shapemeter_columns = get_shapemeter_columns(df)
    feature_signals = [
        resolved_signal
        for candidates in PROCESS_SIGNAL_CANDIDATES
        if (resolved_signal := _resolve_column_name(df.columns, candidates)) is not None
    ] + shapemeter_columns

    rows: list[dict[str, object]] = []
    for coil in coils.itertuples(index=False):
        segment = df.iloc[int(coil.start_idx) : int(coil.end_idx) + 1]
        startup_segment = segment.iloc[:startup_rows]
        stable_segment = segment.iloc[startup_rows:] if len(segment) > startup_rows else segment.iloc[-1:]
        row: dict[str, object] = {
            'coil_id': int(coil.coil_id),
            'start_time': coil.start_time,
            'duration_min': float(coil.duration_min),
            'stitch': coil.stitch,
            'einlaufdicke_um': coil.einlaufdicke_um,
            'solldicke_um': coil.solldicke_um,
            'bandbreite_mm': coil.bandbreite_mm,
        }

        if coil.einlaufdicke_um and coil.solldicke_um:
            row['reduction_pct'] = (1.0 - coil.solldicke_um / coil.einlaufdicke_um) * 100.0

        for signal in feature_signals:
            name = sanitize_feature_name(signal)
            full_values = pd.to_numeric(segment[signal], errors='coerce').dropna()
            startup_values = pd.to_numeric(startup_segment[signal], errors='coerce').dropna()
            stable_values = pd.to_numeric(stable_segment[signal], errors='coerce').dropna()
            if full_values.empty:
                continue

            row[f'{name}_mean'] = float(full_values.mean())
            row[f'{name}_std'] = _safe_std(full_values)
            row[f'{name}_max'] = float(full_values.max())
            row[f'{name}_min'] = float(full_values.min())
            if not startup_values.empty:
                row[f'{name}_startup_mean'] = float(startup_values.mean())
                row[f'{name}_startup_std'] = _safe_std(startup_values)
            if not stable_values.empty:
                row[f'{name}_stable_mean'] = float(stable_values.mean())
                row[f'{name}_stable_std'] = _safe_std(stable_values)

        thickness_values = _numeric_series(segment, 'Dickenabweichung')
        thickness_sigma = _safe_std(thickness_values) if not thickness_values.empty else None
        max_abs_thickness = _safe_max_abs(thickness_values) if not thickness_values.empty else None
        share_abs_gt_1pct = _share_abs_gt(thickness_values, 1.0) if not thickness_values.empty else None
        share_abs_gt_2pct = _share_abs_gt(thickness_values, 2.0) if not thickness_values.empty else None
        count_abs_gt_1pct = int(thickness_values.abs().gt(1.0).sum()) if not thickness_values.empty else 0
        count_abs_gt_2pct = int(thickness_values.abs().gt(2.0).sum()) if not thickness_values.empty else 0

        local_quality_metrics = _build_local_quality_metrics(
            segment=segment,
            interval=interval,
            time_column=time_column,
            shape_columns=shapemeter_columns,
        )
        row.update(local_quality_metrics)

        (
            quality_text,
            quality_label,
            attention_label,
            quality_rank,
            threshold,
            quality_score,
            quality_flags,
            quality_rule,
            quality_detail,
        ) = _compute_coil_label(
            thickness_sigma,
            coil.stitch,
            max_abs_thickness,
            share_abs_gt_2pct,
            row.get('local_peak_sigma_pct'),
            row.get('local_peak_share_abs_gt_2pct'),
            row.get('local_edge_center_shift'),
            row.get('local_quality_window_seconds'),
        )
        row['thickness_sigma_pct'] = thickness_sigma
        row['max_abs_thickness_pct'] = max_abs_thickness
        row['share_abs_gt_1pct'] = share_abs_gt_1pct
        row['share_abs_gt_2pct'] = share_abs_gt_2pct
        row['count_abs_gt_1pct'] = count_abs_gt_1pct
        row['count_abs_gt_2pct'] = count_abs_gt_2pct
        row['quality_threshold_pct'] = threshold
        row['quality_score'] = quality_score
        row['quality_flags'] = quality_flags
        row['quality_rule'] = quality_rule
        row['quality_detail'] = quality_detail
        row['quality'] = quality_text
        row['quality_label'] = quality_label
        row['attention_label'] = attention_label
        row['quality_rank'] = quality_rank
        rows.append(row)

    return pd.DataFrame(rows)



def _build_quality_by_stitch(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    valid = feature_matrix.dropna(subset=['quality_label', 'stitch']).copy()
    if valid.empty:
        return pd.DataFrame(columns=['stitch', 'total_coils', 'bad_coils', 'bad_rate_pct'])

    grouped = valid.groupby('stitch').agg(
        total_coils=('coil_id', 'count'),
        bad_coils=('quality_label', 'sum'),
    )
    grouped['bad_rate_pct'] = grouped['bad_coils'] / grouped['total_coils'] * 100.0
    return grouped.reset_index().sort_values('stitch')



def _candidate_feature_columns(feature_matrix: pd.DataFrame) -> list[str]:
    numeric_columns = feature_matrix.select_dtypes(include=[np.number]).columns.tolist()
    excluded = {
        'coil_id',
        'quality_label',
        'attention_label',
        'quality_rank',
        'quality_score',
        'quality_threshold_pct',
        'thickness_sigma_pct',
        'max_abs_thickness_pct',
        'share_abs_gt_1pct',
        'share_abs_gt_2pct',
        'count_abs_gt_1pct',
        'count_abs_gt_2pct',
        'baseline_sigma_pct',
        'local_quality_window_seconds',
        'local_peak_sigma_pct',
        'local_peak_mean_abs_thickness_pct',
        'local_peak_q95_abs_thickness_pct',
        'local_peak_max_abs_thickness_pct',
        'local_peak_share_abs_gt_2pct',
        'local_edge_center_shift',
    }
    candidates = [column for column in numeric_columns if column not in excluded]
    return [column for column in candidates if not column.startswith(LEAKAGE_PREFIXES)]



def _top_correlations(feature_matrix: pd.DataFrame, feature_columns: list[str], top_n: int = 15) -> pd.DataFrame:
    if not feature_columns:
        return pd.DataFrame(columns=['feature', 'correlation', 'abs_correlation'])

    numeric = feature_matrix[feature_columns + ['quality_label']].dropna(subset=['quality_label']).copy()
    if numeric.empty or numeric['quality_label'].nunique(dropna=True) < 2:
        return pd.DataFrame(columns=['feature', 'correlation', 'abs_correlation'])

    correlations: list[dict[str, float | str]] = []
    for column in feature_columns:
        pair = numeric[[column, 'quality_label']].dropna()
        if len(pair) < 3:
            continue
        if pair[column].nunique(dropna=True) < 2 or pair['quality_label'].nunique(dropna=True) < 2:
            continue
        corr = pair[column].corr(pair['quality_label'])
        if pd.notna(corr):
            correlations.append(
                {
                    'feature': column,
                    'correlation': float(corr),
                    'abs_correlation': float(abs(corr)),
                }
            )
    if not correlations:
        return pd.DataFrame(columns=['feature', 'correlation', 'abs_correlation'])
    return pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False).head(top_n)



def run_prediction(
    feature_matrix: pd.DataFrame,
    prediction_dir: Path,
    rf_params: dict[str, object] | None = None,
) -> PredictionArtifacts:
    prediction_dir.mkdir(parents=True, exist_ok=True)

    if feature_matrix.empty:
        empty = pd.DataFrame()
        return PredictionArtifacts(empty, empty, empty, empty, None, None, rf_params or {})

    feature_matrix.to_csv(prediction_dir / 'feature_matrix.csv', sep=';', decimal=',', index=False)
    quality_by_stitch = _build_quality_by_stitch(feature_matrix)
    quality_by_stitch.to_csv(prediction_dir / 'quality_by_stitch.csv', sep=';', decimal=',', index=False)

    valid = feature_matrix.dropna(subset=['quality_label']).copy()
    feature_columns = _candidate_feature_columns(valid)
    top_correlations = _top_correlations(valid, feature_columns, top_n=20)
    top_correlations.to_csv(prediction_dir / 'top_correlations.csv', sep=';', decimal=',', index=False)

    if valid.empty or valid['quality_label'].nunique() < 2 or len(valid) < 8 or not feature_columns:
        return PredictionArtifacts(
            feature_matrix=feature_matrix,
            quality_by_stitch=quality_by_stitch,
            top_correlations=top_correlations,
            top_rf_features=pd.DataFrame(),
            accuracy_mean=None,
            accuracy_std=None,
            rf_params=rf_params or {},
        )

    X = valid[feature_columns]
    y = valid['quality_label'].astype(int)
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    params = dict(RANDOM_FOREST_DEFAULTS)
    if rf_params:
        params.update(rf_params)
    model = RandomForestClassifier(**params)

    min_class_size = int(y.value_counts().min())
    n_splits = min(5, min_class_size)
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_imputed, y, cv=cv, scoring='accuracy')
        accuracy_mean = float(scores.mean())
        accuracy_std = float(scores.std())
    else:
        accuracy_mean = None
        accuracy_std = None

    model.fit(X_imputed, y)
    top_rf_features = (
        pd.DataFrame({'feature': feature_columns, 'importance': model.feature_importances_})
        .sort_values('importance', ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    top_rf_features.to_csv(prediction_dir / 'rf_importance.csv', sep=';', decimal=',', index=False)

    if not top_rf_features.empty:
        figure = px.bar(
            top_rf_features.head(15),
            x='importance',
            y='feature',
            orientation='h',
            title='Random Forest Feature Importance',
        )
        figure.update_layout(yaxis={'categoryorder': 'total ascending'})
        figure.write_html(prediction_dir / 'rf_importance.html', include_plotlyjs='cdn')

    if not quality_by_stitch.empty:
        figure = px.bar(
            quality_by_stitch,
            x='stitch',
            y='bad_rate_pct',
            title='Fehlerquote pro Stich',
            text='bad_rate_pct',
        )
        figure.write_html(prediction_dir / 'quality_by_stitch.html', include_plotlyjs='cdn')

    if not top_correlations.empty:
        figure = px.bar(
            top_correlations.head(15),
            x='abs_correlation',
            y='feature',
            orientation='h',
            title='Top Korrelationen mit dem Qualitaetslabel',
        )
        figure.update_layout(yaxis={'categoryorder': 'total ascending'})
        figure.write_html(prediction_dir / 'top_correlations.html', include_plotlyjs='cdn')

    return PredictionArtifacts(
        feature_matrix=feature_matrix,
        quality_by_stitch=quality_by_stitch,
        top_correlations=top_correlations,
        top_rf_features=top_rf_features,
        accuracy_mean=accuracy_mean,
        accuracy_std=accuracy_std,
        rf_params=params,
    )
