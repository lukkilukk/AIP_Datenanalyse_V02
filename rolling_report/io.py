from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class DatasetMeta:
    file_path: Path
    time_column: str
    n_rows: int
    n_columns: int
    base_signal_count: int
    step_seconds: float | None
    start_time: Any
    end_time: Any
    units_by_column: dict[str, str]

    @property
    def duration_hours(self) -> float | None:
        if self.start_time is None or self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return delta.total_seconds() / 3600.0


def _read_header_rows(file_path: Path) -> tuple[list[str], list[str], list[str]]:
    with file_path.open('r', encoding='latin-1', errors='replace') as handle:
        ids = handle.readline().rstrip('\n').split(';')
        names = handle.readline().rstrip('\n').split(';')
        units = handle.readline().rstrip('\n').split(';')
    return ids, names, units


def _make_unique_names(preferred: list[str], fallback: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result: list[str] = []
    for index, (name, backup) in enumerate(zip(preferred, fallback)):
        label = (name or '').strip() or (backup or '').strip() or f'col_{index}'
        if label in seen:
            seen[label] += 1
            label = f'{label}__{seen[label]}'
        else:
            seen[label] = 0
        result.append(label)
    return result


def load_measurement_file(file_path: str | Path) -> tuple[pd.DataFrame, DatasetMeta]:
    file_path = Path(file_path)
    ids, signal_names, units = _read_header_rows(file_path)
    columns = _make_unique_names(signal_names, ids)

    df = pd.read_csv(
        file_path,
        sep=';',
        skiprows=3,
        header=None,
        names=columns,
        encoding='latin-1',
        decimal=',',
        low_memory=False,
    )

    time_column = columns[0]
    df[time_column] = pd.to_datetime(df[time_column], errors='coerce')

    if len(columns) > 1:
        second_column = columns[1]
        if second_column.endswith('_text') or second_column == '0:0':
            df[second_column] = df[second_column].astype('string')

    start_time = df[time_column].dropna().iloc[0] if df[time_column].notna().any() else None
    end_time = df[time_column].dropna().iloc[-1] if df[time_column].notna().any() else None
    step_seconds = None
    if df[time_column].notna().sum() >= 2:
        step = df[time_column].dropna().diff().dt.total_seconds().median()
        if pd.notna(step):
            step_seconds = float(step)

    base_signal_count = sum(
        1 for column in columns if not column.endswith('.max') and not column.endswith('.min')
    )
    units_by_column = {column: units[index] if index < len(units) else '' for index, column in enumerate(columns)}

    meta = DatasetMeta(
        file_path=file_path,
        time_column=time_column,
        n_rows=len(df),
        n_columns=len(df.columns),
        base_signal_count=base_signal_count,
        step_seconds=step_seconds,
        start_time=start_time,
        end_time=end_time,
        units_by_column=units_by_column,
    )
    return df, meta
