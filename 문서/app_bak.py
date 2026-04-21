import os
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text

# CHANGELOG
# - Structured RCA storytelling (Scope → Loss Structure → Machine Concentration → Stop Reason Analysis → Quality Association → Loss Path Priority → Action Recommendation → RCA Summary) with numeric evidence, observational wording, and limitation notes.
# - Added reliability badges, micro/macro stop characterization, quality/load association, and sample-data safeguards.
# - Dual RCA summary generation (Executive + Operator) with action templates and data quality gates.


# ------------------------------------------------------------------
# Theme & constants
# ------------------------------------------------------------------
DARK_TEMPLATE = 'plotly_dark'
KPI_COLORS = {'good': '#00CC96', 'medium': '#FFA15A', 'bad': '#EF553B'}
CARD_BG = '#11151d'
CARD_TEXT = '#fff'
INFO_TEXT = '#b0b0b0'
PRIMARY_COLOR = '#1f77b4'
SECONDARY_COLOR = '#ff7f0e'
STOP_REASON_GROUP_COLORS = {
    'Process': '#636efa',
    'Quality': '#ef553b',
    'Equipment': '#00cc96',
    'Waiting': '#ab63fa',
    'Default': '#ffa15a'
}
ACTION_TEMPLATES = {
    'CHANGEOVER': [
        '셋업/티칭/피더 세팅 체크리스트 적용, 초물(First Article) 확인 강화',
        'LOT 시작 직후(예: 30분) ERROR 4종 증가 여부 비교'
    ],
    'PICKUP_ERR': [
        '피더(feeder_id)·노즐(nozzle_serial) Hotspot 점검(테이프/스플라이스/진공/노즐 오염)',
        '에러 상위 부품(part_number) 릴/공급 문제 점검'
    ],
    'RECOG_ERR': [
        '조명/카메라 오염, 마크 인식 티칭, 보드 휨/마크 품질 점검',
        '특정 LOT/시퀀스 구간 집중 여부 확인'
    ],
    'PLACE_ERR': [
        '헤드/축, 프로그램 오프셋, 라이브러리(library_name), 부품 높이/좌표 점검',
        '특정 노즐/피더/부품 조합 편중 확인'
    ],
    'TRANSFER_ERR': [
        '컨베이어/버퍼/센서/인터록, 레일 폭/클램프 점검',
        'WAIT_PRE/WAIT_POST 동시 증가 여부 확인'
    ],
    'WAIT_PRE': [
        '전후공정 병목/라인 밸런스/버퍼 Full/인터록 조건 점검',
        '대기 집중 stage/machine을 Loss Tree/랭킹으로 확인'
    ],
    'WAIT_POST': [
        '전후공정 병목/라인 밸런스/버퍼 Full/인터록 조건 점검',
        '대기 집중 stage/machine을 Loss Tree/랭킹으로 확인'
    ]
}

POC_BADGE_TEXT = 'POC 관측 기반 · 시간 근사/누적 proxy 가능'
RELIABILITY_BADGE_TITLE = '이벤트형/누적형(추정)'
RELIABILITY_THRESHOLD = 0.5


# ------------------------------------------------------------------
# DB helpers
# ------------------------------------------------------------------
@st.cache_resource(ttl=3600)
def get_engine():
    user = os.environ.get('PGUSER', 'analysis')
    password = os.environ.get('PGPASSWORD', 'analysis1!')
    host = os.environ.get('PGHOST', '192.168.200.105')
    port = os.environ.get('PGPORT', '5432')
    database = os.environ.get('PGDATABASE', 'nexedge')
    url = f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}'
    return create_engine(url, pool_pre_ping=True)


@st.cache_data(ttl=3600)
def load_data() -> Dict[str, pd.DataFrame]:
    engine = get_engine()

    sql_map = {
        'machine_time_summary': '''
            SELECT machine_id, lot_id, line_id, stage_no, machine_order,
                   running_time_sec, real_running_time_sec, power_on_time_sec,
                   total_stop_time_sec, stop_count, transfer_time_sec,
                   board_recognition_time_sec, placement_time_sec, recorded_at
            FROM machine_time_summary
            WHERE recorded_at >= current_date - interval '30 days'
        ''',
        'stop_log': '''
            SELECT stop_reason_code, lot_machine_id, duration_sec, stop_count,
                   recorded_at, source_file_id
            FROM stop_log
            WHERE recorded_at >= current_date - interval '30 days'
            ORDER BY source_file_id DESC
            LIMIT 2000
        ''',
        'stop_reason': 'SELECT stop_reason_code, stop_reason_name, stop_reason_group FROM stop_reason',
        'lot_machine': 'SELECT lot_machine_id, lot_id, machine_id, line_id, stage_no, machine_order FROM lot_machine',
        'machine': 'SELECT machine_id, machine_name, line_id, stage_no, machine_order FROM machine',
        'lot': 'SELECT lot_id, lot_name, line_id FROM lot',
        'file': 'SELECT file_id, file_datetime, file_sequence FROM file',
        'pickup_error_summary': '''
            SELECT machine_id, lot_id, line_id, stage_no,
                   pickup_count, error_count, pickup_error_count, recognition_error_count
            FROM pickup_error_summary
            WHERE recorded_at >= current_date - interval '30 days'
        ''',
        'component_pickup_summary': '''
            SELECT component_id, machine_id, lot_id,
                   pickup_count, error_count, pickup_error_count, recognition_error_count,
                   defect_type, recorded_at
            FROM component_pickup_summary
            WHERE recorded_at >= current_date - interval '30 days'
            ORDER BY recorded_at DESC
            LIMIT 2000
        ''',
        'component': 'SELECT component_id, part_number, library_name, feeder_id, feeder_serial, nozzle_serial FROM component'
    }
    data: Dict[str, pd.DataFrame] = {}
    with engine.connect() as conn:
        for name, query in sql_map.items():
            data[name] = pd.read_sql(text(query), conn)
    data = normalize_datetimes(data)
    data['_meta'] = {'is_sample': False}
    return data


def normalize_datetimes(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    for key in ['machine_time_summary', 'component_pickup_summary', 'stop_log']:
        df = data.get(key)
        if df is not None and 'recorded_at' in df.columns:
            df['recorded_at'] = pd.to_datetime(df['recorded_at'], errors='coerce')
    file_df = data.get('file')
    if file_df is not None and 'file_datetime' in file_df.columns:
        file_df['file_datetime'] = pd.to_datetime(file_df['file_datetime'], errors='coerce')
    return data


@st.cache_data(ttl=3600)
def generate_sample_data() -> Dict[str, pd.DataFrame]:
    base = datetime.now()
    lines = ['Line-A', 'Line-B', 'Line-C']
    machines = [f'M{idx:02d}' for idx in range(1, 10)]
    lots = [f'LOT{idx:03d}' for idx in range(1, 7)]

    machine_time_rows = []
    lot_machine_rows = []
    for idx, machine_id in enumerate(machines):
        line_id = lines[idx % len(lines)]
        stage_no = (idx % 3) + 1
        lot_id = lots[idx % len(lots)]
        lot_machine_id = idx + 1
        machine_time_rows.append({
            'machine_id': machine_id,
            'lot_id': lot_id,
            'line_id': line_id,
            'stage_no': stage_no,
            'machine_order': idx + 1,
            'running_time_sec': 3600 + idx * 90,
            'real_running_time_sec': 3200 + idx * 80,
            'power_on_time_sec': 4000 + idx * 70,
            'total_stop_time_sec': 200 + idx * 12,
            'stop_count': 4 + (idx % 4),
            'transfer_time_sec': 120 + idx * 5,
            'board_recognition_time_sec': 40 + idx * 2,
            'placement_time_sec': 60 + idx * 3,
            'recorded_at': base - timedelta(days=idx)
        })
        lot_machine_rows.append({
            'lot_machine_id': lot_machine_id,
            'lot_id': lot_id,
            'machine_id': machine_id,
            'line_id': line_id,
            'stage_no': stage_no,
            'machine_order': idx + 1
        })

    stop_reasons = [
        ('CHANGEOVER', 'Changeover', 'Process'),
        ('PICKUP_ERR', 'Pickup Error', 'Quality'),
        ('PLACE_ERR', 'Placement Error', 'Equipment'),
        ('RECOG_ERR', 'Recognition Error', 'Quality'),
        ('TRANSFER_ERR', 'Transfer Error', 'Equipment'),
        ('WAIT_PRE', 'Wait Pre', 'Waiting'),
        ('WAIT_POST', 'Wait Post', 'Waiting')
    ]

    file_rows = []
    for idx in range(1, 7):
        file_rows.append({
            'file_id': idx,
            'file_datetime': base - timedelta(days=idx // 2, hours=idx * 3),
            'file_sequence': idx * 10
        })

    stop_log_rows = []
    for idx in range(12):
        reason = stop_reasons[idx % len(stop_reasons)]
        lot_machine = lot_machine_rows[idx % len(lot_machine_rows)]
        stop_log_rows.append({
            'stop_reason_code': reason[0],
            'lot_machine_id': lot_machine['lot_machine_id'],
            'duration_sec': 45 + idx * 5,
            'stop_count': 1 if idx % 3 else 3,
            'recorded_at': base - timedelta(hours=idx * 2),
            'source_file_id': file_rows[idx % len(file_rows)]['file_id']
        })

    pickup_error_rows = []
    for idx, machine in enumerate(machines[:6]):
        lot_id = lots[idx % len(lots)]
        pickup_error_rows.append({
            'machine_id': machine,
            'lot_id': lot_id,
            'line_id': lines[idx % len(lines)],
            'stage_no': (idx % 3) + 1,
            'pickup_count': 1000 + idx * 40,
            'error_count': 10 + idx * 2,
            'pickup_error_count': 7 + idx,
            'recognition_error_count': 3 + (idx % 3)
        })

    component_rows = []
    component_pickup_rows = []
    for idx in range(12):
        component_id = f'C{idx:03d}'
        machine_id = machines[idx % len(machines)]
        lot_id = lots[idx % len(lots)]
        component_rows.append({
            'component_id': component_id,
            'part_number': f'PN-{idx:04d}',
            'library_name': f'LIB-{(idx % 3) + 1}',
            'feeder_id': f'FDR-{(idx % 5) + 1}',
            'feeder_serial': f'SN{idx + 100}',
            'nozzle_serial': f'NOZ{idx + 200}'
        })
        component_pickup_rows.append({
            'component_id': component_id,
            'machine_id': machine_id,
            'lot_id': lot_id,
            'pickup_count': 800 + idx * 5,
            'error_count': (idx % 5) + 1,
            'pickup_error_count': (idx % 3) + 1,
            'recognition_error_count': (idx % 2),
            'defect_type': f'Defect-{(idx % 4) + 1}',
            'recorded_at': base - timedelta(hours=idx)
        })

    return normalize_datetimes({
        'machine_time_summary': pd.DataFrame(machine_time_rows),
        'stop_log': pd.DataFrame(stop_log_rows),
        'stop_reason': pd.DataFrame(stop_reasons, columns=['stop_reason_code', 'stop_reason_name', 'stop_reason_group']),
        'lot_machine': pd.DataFrame(lot_machine_rows),
        'machine': pd.DataFrame([{
            'machine_id': row['machine_id'],
            'machine_name': f'Machine {row["machine_id"]}',
            'line_id': row['line_id'],
            'stage_no': row['stage_no'],
            'machine_order': row['machine_order']
        } for row in lot_machine_rows]),
        'lot': pd.DataFrame({
            'lot_id': lots,
            'lot_name': [f'LOT_NAME_{idx}' for idx in range(1, len(lots) + 1)],
            'line_id': [lines[idx % len(lines)] for idx in range(len(lots))]
        }),
        'file': pd.DataFrame(file_rows),
        'pickup_error_summary': pd.DataFrame(pickup_error_rows),
        'component_pickup_summary': pd.DataFrame(component_pickup_rows),
        'component': pd.DataFrame(component_rows),
        '_meta': {'is_sample': True}
    })


# ------------------------------------------------------------------
# Data modeling helpers
# ------------------------------------------------------------------
@st.cache_data(ttl=3600)
def build_views(raw_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    stop_log = (raw_dfs.get('stop_log', pd.DataFrame()).copy())
    stop_reason = raw_dfs.get('stop_reason', pd.DataFrame())
    lot_machine = raw_dfs.get('lot_machine', pd.DataFrame())
    machine = raw_dfs.get('machine', pd.DataFrame())
    lot = raw_dfs.get('lot', pd.DataFrame())
    file_df = raw_dfs.get('file', pd.DataFrame())

    stop_log['duration_sec'] = pd.to_numeric(stop_log.get('duration_sec', 0), errors='coerce').fillna(0)
    stop_log['stop_count'] = pd.to_numeric(stop_log.get('stop_count', 1), errors='coerce').fillna(1)

    stop_enriched = (
        stop_log
        .merge(stop_reason, on='stop_reason_code', how='left')
        .merge(lot_machine, on='lot_machine_id', how='left')
        .merge(machine[['machine_id', 'machine_order']], on='machine_id', how='left')
        .merge(lot[['lot_id', 'lot_name']], on='lot_id', how='left')
        .merge(file_df[['file_id', 'file_datetime', 'file_sequence']], left_on='source_file_id', right_on='file_id', how='left')
    )

    stop_enriched['file_datetime'] = pd.to_datetime(stop_enriched['file_datetime'], errors='coerce')
    stop_enriched['day'] = stop_enriched['file_datetime'].dt.date
    stop_enriched['time_axis_is_approx'] = stop_enriched['file_datetime'].dt.time == datetime.min.time()
    stop_enriched['time_sort'] = stop_enriched['file_datetime'] + pd.to_timedelta(stop_enriched['file_sequence'].fillna(0), unit='s')
    stop_enriched['time_label'] = stop_enriched.apply(
        lambda row: row['file_datetime'].strftime('%Y-%m-%d %H:%M:%S') if pd.notnull(row['file_datetime']) else f"Seq-{int(row['file_sequence'] or 0)}",
        axis=1
    )
    dup_mask = stop_enriched.duplicated(subset=['lot_machine_id', 'stop_reason_code'], keep=False)
    stop_enriched['is_aggregated'] = (stop_enriched['stop_count'] > 1) | dup_mask
    stop_enriched['avg_stop_duration_sec'] = np.where(
        stop_enriched['is_aggregated'],
        stop_enriched['duration_sec'] / stop_enriched['stop_count'].replace({0: 1}),
        stop_enriched['duration_sec']
    )

    def _coalesce_first(df: pd.DataFrame, cols: list[str], default=np.nan) -> pd.Series:
        out = None
        for c in cols:
            if c in df.columns:
               out = df[c] if out is None else out.combine_first(df[c])
        if out is None:
            return pd.Series([default] * len(df), index=df.index)
        return out

    stop_enriched["machine_order"] = _coalesce_first(
    stop_enriched,
    ["machine_order", "machine_order_x", "machine_order_y"],
    default=0 )

    stop_enriched['line_id'] = stop_enriched['line_id'].fillna('Unknown')
    stop_enriched['stage_no'] = stop_enriched['stage_no'].fillna(-1).astype(int)
    stop_enriched['machine_order'] = stop_enriched['machine_order'].fillna(0).astype(int)

    stop_enriched = stop_enriched.rename(columns={'stop_reason_name': 'stop_reason_name', 'stop_reason_group': 'stop_reason_group'})

    machine_time = raw_dfs.get('machine_time_summary', pd.DataFrame()).copy()
    pickup_error = raw_dfs.get('pickup_error_summary', pd.DataFrame()).copy()

    machine_time['running_time_sec'] = pd.to_numeric(machine_time.get('running_time_sec', 0), errors='coerce').fillna(0)
    machine_time['power_on_time_sec'] = pd.to_numeric(machine_time.get('power_on_time_sec', 0), errors='coerce').fillna(0)
    machine_time['real_running_time_sec'] = pd.to_numeric(machine_time.get('real_running_time_sec', 0), errors='coerce').fillna(0)
    machine_time['total_stop_time_sec'] = pd.to_numeric(machine_time.get('total_stop_time_sec', 0), errors='coerce').fillna(0)
    machine_time['transfer_time_sec'] = pd.to_numeric(machine_time.get('transfer_time_sec', 0), errors='coerce').fillna(0)
    machine_time['board_recognition_time_sec'] = pd.to_numeric(machine_time.get('board_recognition_time_sec', 0), errors='coerce').fillna(0)
    machine_time['placement_time_sec'] = pd.to_numeric(machine_time.get('placement_time_sec', 0), errors='coerce').fillna(0)
    machine_time['stop_count'] = pd.to_numeric(machine_time.get('stop_count', 0), errors='coerce').fillna(0)
    machine_time['day'] = pd.to_datetime(machine_time.get('recorded_at')).dt.date

    pickup_error['pickup_count'] = pd.to_numeric(pickup_error.get('pickup_count', 0), errors='coerce').fillna(0)
    pickup_error['error_count'] = pd.to_numeric(pickup_error.get('error_count', 0), errors='coerce').fillna(0)

    machine_summary = (
        machine_time
        .groupby(['line_id', 'stage_no', 'machine_id', 'machine_order', 'lot_id', 'day'], as_index=False)
        .agg({
            'running_time_sec': 'sum',
            'real_running_time_sec': 'sum',
            'power_on_time_sec': 'sum',
            'total_stop_time_sec': 'sum',
            'stop_count': 'sum',
            'transfer_time_sec': 'sum',
            'board_recognition_time_sec': 'sum',
            'placement_time_sec': 'sum'
        })
    )

    pickup_agg = (
        pickup_error
        .groupby(['line_id', 'stage_no', 'machine_id', 'lot_id'], as_index=False)
        .agg({
            'pickup_count': 'sum',
            'error_count': 'sum'
        })
        .rename(columns={'pickup_count': 'total_pickup_count', 'error_count': 'total_error_count'})
    )

    machine_summary = (
        machine_summary
        .merge(pickup_agg, on=['line_id', 'stage_no', 'machine_id', 'lot_id'], how='left')
        .merge(lot[['lot_id', 'lot_name']], on='lot_id', how='left')
    )

    machine_summary['total_pickup_count'] = machine_summary['total_pickup_count'].fillna(0)
    machine_summary['total_error_count'] = machine_summary['total_error_count'].fillna(0)

    machine_summary['availability'] = machine_summary.apply(
        lambda row: safe_div(row['running_time_sec'], row['power_on_time_sec']), axis=1
    )
    machine_summary['performance'] = machine_summary.apply(
        lambda row: safe_div(row['real_running_time_sec'], row['running_time_sec']), axis=1
    )
    machine_summary['quality'] = machine_summary.apply(
        lambda row: safe_div(row['total_pickup_count'] - row['total_error_count'], row['total_pickup_count']), axis=1
    )
    machine_summary['oee'] = machine_summary['availability'] * machine_summary['performance'] * machine_summary['quality']

    component_pickup = raw_dfs.get('component_pickup_summary', pd.DataFrame()).copy()
    component_pickup['pickup_count'] = pd.to_numeric(component_pickup.get('pickup_count', 0), errors='coerce').fillna(0)
    component_pickup['error_count'] = pd.to_numeric(component_pickup.get('error_count', 0), errors='coerce').fillna(0)
    component_pickup['pickup_error_count'] = pd.to_numeric(component_pickup.get('pickup_error_count', 0), errors='coerce').fillna(0)
    component_pickup['recognition_error_count'] = pd.to_numeric(component_pickup.get('recognition_error_count', 0), errors='coerce').fillna(0)
    component_pickup['recorded_at'] = pd.to_datetime(component_pickup.get('recorded_at'), errors='coerce')
    component_pickup['day'] = component_pickup['recorded_at'].dt.date

    component = raw_dfs.get('component', pd.DataFrame())
    component_quality = (
        component_pickup
        .merge(component, on='component_id', how='left')
        .merge(machine[['machine_id', 'machine_order', 'line_id', 'stage_no']], on='machine_id', how='left')
        .merge(lot[['lot_id', 'lot_name']], on='lot_id', how='left')
    )

    component_quality['error_rate'] = component_quality.apply(
        lambda row: safe_div(row['error_count'], row['pickup_count']), axis=1
    )

    return {
        'vw_stop_enriched': stop_enriched,
        'vw_lot_machine_summary': machine_summary,
        'vw_component_quality': component_quality
    }


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------
def safe_div(a, b):
    try:
        if b is None or pd.isna(b) or float(b) == 0:
            return 0
        return float(a) / float(b)
    except Exception:
        return 0


def format_duration(seconds: float) -> str:
    try:
        seconds = int(seconds)
        return str(timedelta(seconds=seconds))
    except Exception:
        return '0:00:00'


def apply_filters(df: pd.DataFrame, filters: Dict[str, List[str]], name: str):
    if df.empty:
        return df
    filtered = df.copy()
    if 'day' in filtered.columns and filters.get('date_range'):
        start_date, end_date = filters['date_range']
        filtered = filtered[filtered['day'].between(start_date, end_date)]
    for column, key in [('line_id', 'lines'), ('stage_no', 'stages'), ('machine_id', 'machines'), ('lot_id', 'lots')]:
        if column in filtered.columns and filters.get(key):
            values = filters[key]
            if values:
                filtered = filtered[filtered[column].isin(values)]
    if name == 'vw_stop_enriched':
        min_stop = filters.get('min_stop_count', 0)
        if min_stop:
            filtered = filtered[filtered['stop_count'] >= min_stop]
    if name == 'vw_component_quality':
        min_pickup = filters.get('min_pickup_count', 0)
        if min_pickup:
            filtered = filtered[filtered['pickup_count'] >= min_pickup]
    return filtered


def collect_filters(views: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    stop_df = views['vw_stop_enriched']
    machine_df = views['vw_lot_machine_summary']

    st.sidebar.markdown('### Filters')
    default_dates = (date.today() - timedelta(days=6), date.today())
    if not stop_df.empty and 'day' in stop_df.columns:
        min_day = stop_df['day'].min()
        max_day = stop_df['day'].max()
        if pd.notnull(min_day) and pd.notnull(max_day):
            default_dates = (max(min_day, max_day - timedelta(days=6)), max_day)
    date_range = st.sidebar.date_input('Date Range', value=default_dates)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = default_dates

    line_options = sorted(set(stop_df['line_id'].dropna().tolist() + machine_df['line_id'].dropna().tolist()))
    stage_options = sorted(machine_df['stage_no'].dropna().unique())
    machine_options = sorted(machine_df['machine_id'].dropna().unique())
    lot_options = sorted(machine_df['lot_id'].dropna().unique())

    line_selection = st.sidebar.multiselect('Line', line_options, default=line_options)
    stage_selection = st.sidebar.multiselect('Stage', stage_options, default=stage_options)
    machine_selection = st.sidebar.multiselect('Machine', machine_options, default=machine_options)
    lot_selection = st.sidebar.multiselect('Lot', lot_options, default=lot_options)
    min_stop_count = st.sidebar.slider('Min Stop Count', min_value=1, max_value=50, value=1)
    min_pickup_count = st.sidebar.slider('Min Pickup Count', min_value=10, max_value=400, value=10)

    return {
        'date_range': (start_date, end_date),
        'lines': line_selection,
        'stages': stage_selection,
        'machines': machine_selection,
        'lots': lot_selection,
        'min_stop_count': min_stop_count,
        'min_pickup_count': min_pickup_count
    }


def calculate_kpis(vw_lot_machine_summary: pd.DataFrame, vw_stop_enriched: pd.DataFrame) -> Dict[str, float]:
    if vw_lot_machine_summary.empty:
        return {'oee': 0, 'availability': 0, 'performance': 0, 'quality': 0, 'mtbf': 0, 'mttr': 0}

    running_sum = vw_lot_machine_summary['running_time_sec'].sum()
    power_on_sum = vw_lot_machine_summary['power_on_time_sec'].sum()
    real_running_sum = vw_lot_machine_summary['real_running_time_sec'].sum()
    stop_time_sum = vw_lot_machine_summary['total_stop_time_sec'].sum()
    stop_count = vw_lot_machine_summary['stop_count'].sum()

    pickup_sum = vw_lot_machine_summary['total_pickup_count'].sum()
    error_sum = vw_lot_machine_summary['total_error_count'].sum()

    availability = safe_div(running_sum, power_on_sum)
    performance = safe_div(real_running_sum, running_sum)
    quality = safe_div(pickup_sum - error_sum, pickup_sum)
    oee = availability * performance * quality
    mtbf = safe_div(running_sum, stop_count)
    mttr = safe_div(stop_time_sum, stop_count)

    return {
        'oee': oee,
        'availability': availability,
        'performance': performance,
        'quality': quality,
        'mtbf': mtbf,
        'mttr': mttr
    }


def render_kpi_cards(kpis: Dict[str, float]):
    cols = st.columns(4)
    has_data = any(kpis.get(label) for label in ['oee', 'availability', 'quality', 'mtbf'])
    formatted = {
        'oee': f"{kpis.get('oee', 0) * 100:.1f}%" if has_data else '-',
        'availability': f"{kpis.get('availability', 0) * 100:.1f}%" if has_data else '-',
        'quality': f"{kpis.get('quality', 0) * 100:.1f}%" if has_data else '-',
        'mtbf': f"{kpis.get('mtbf', 0):,.0f}s" if has_data else '-'
    }

    oee_color = KPI_COLORS['bad']
    oee_value = kpis.get('oee', 0)
    if oee_value >= 0.85:
        oee_color = KPI_COLORS['good']
    elif oee_value >= 0.7:
        oee_color = KPI_COLORS['medium']

    card_specs = [
        {'label': 'OEE', 'value': formatted['oee'], 'bg': oee_color},
        {'label': 'Availability', 'value': formatted['availability'], 'bg': CARD_BG},
        {'label': 'Quality', 'value': formatted['quality'], 'bg': CARD_BG},
        {'label': 'MTBF', 'value': formatted['mtbf'], 'bg': CARD_BG}
    ]

    for col, spec in zip(cols, card_specs):
        col.markdown(f"""
            <div style='background:{spec['bg']}; padding:18px; border-radius:18px; text-align:center; box-shadow:0 8px 20px rgba(0,0,0,.35);'>
                <div style='color:{CARD_TEXT}; font-size:14px; text-transform:uppercase; letter-spacing:1px;'>{spec['label']}</div>
                <div style='color:{CARD_TEXT}; font-size:32px; font-weight:600; margin-top:12px;'>{spec['value']}</div>
            </div>
        """, unsafe_allow_html=True)


# ------------------------------------------------------------------
# UI helpers
# ------------------------------------------------------------------
def render_insights(bullets: List[str]):
    if not bullets:
        return
    st.markdown("""
        <div style='background:#161b24; border-radius:14px; padding:16px; color:#d1d1d1;'>
            <strong style='color:#fff;'>Insight Summary</strong>
            <ul style='margin:8px 0 0 16px; padding:0; list-style:disc;'>
    """, unsafe_allow_html=True)
    for bullet in bullets:
        st.markdown(f"<li>{bullet}</li>", unsafe_allow_html=True)
    st.markdown("""
            </ul>
        </div>
    """, unsafe_allow_html=True)


def render_no_data(reason: str = 'No Data Available'):
    st.markdown(f"""
        <div style='border:1px solid #333; border-radius:18px; padding:24px; text-align:center; background:#111; color:#ccc;'>
            <p style='font-size:18px; margin:4px 0;'>No Data Available</p>
            <small>{reason}</small>
        </div>
    """, unsafe_allow_html=True)


def render_action_templates(stop_codes: List[str]):
    if not stop_codes:
        return
    st.markdown('### Action Templates (Action):')
    for code in stop_codes:
        actions = ACTION_TEMPLATES.get(code, [])
        if not actions:
            continue
        st.markdown(f"**{code}**")
        for act in actions:
            st.markdown(f"- {act}")


def highlight_time_note(is_approx: bool) -> str:
    return ' (파일 시퀀스 기반(시간 근사))' if is_approx else ''


def extract_top_stop_reasons(stop_df: pd.DataFrame) -> List[str]:
    if stop_df.empty:
        return []
    reason_ranking = (
        stop_df.groupby(['stop_reason_code', 'stop_reason_name'])['duration_sec']
        .sum()
        .reset_index()
        .sort_values('duration_sec', ascending=False)
    )
    return reason_ranking['stop_reason_code'].head(3).tolist()


def summarize_filters(filters: Dict[str, List[str]]) -> str:
    parts = []
    lens = [('Line', filters.get('lines')), ('Stage', filters.get('stages')), ('Machine', filters.get('machines')), ('Lot', filters.get('lots'))]
    for label, vals in lens:
        if not vals:
            parts.append('전체')
        elif len(vals) <= 3:
            parts.append(','.join(map(str, vals)))
        else:
            parts.append(f'{vals[0]} 외 {len(vals)-1}개')
    return '; '.join(parts)


def entropy_insight_label(filters: Dict[str, List[str]]) -> str:
    start, end = filters.get('date_range', (date.today(), date.today()))
    return f"기간: {start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')}"


def render_error_rate_table(data: pd.DataFrame):
    idx = data.sort_values('error_rate', ascending=False).head(10)
    idx = idx[['machine_id', 'error_rate', 'error_count', 'pickup_count']]
    idx['error_rate'] = idx['error_rate'].apply(lambda v: f"{v * 100:.1f}%")
    st.table(idx)


def render_sample_size_warning(count: int):
    if count and count < 10:
        st.warning("표본 수 제한으로 해석 주의 필요")


def compute_reliability_indicators(stop_df: pd.DataFrame) -> Dict[str, float]:
    total = len(stop_df)
    if total == 0:
        return {'total': 0, 'aggregated_ratio': 0, 'duplicate_ratio': 0, 'coverage_ratio': 0, 'approx_ratio': 0,
                'aggregated_count': 0, 'duplicate_count': 0, 'coverage_count': 0, 'approx_count': 0}
    aggregated_count = int((stop_df['stop_count'] > 1).sum())
    duplicate_count = int(stop_df.duplicated(subset=['lot_machine_id', 'stop_reason_code', 'machine_id'], keep=False).sum())
    coverage_count = int(stop_df['file_id'].notna().sum())
    approx_count = int(stop_df['time_axis_is_approx'].sum())
    return {
        'total': total,
        'aggregated_ratio': safe_div(aggregated_count, total),
        'duplicate_ratio': safe_div(duplicate_count, total),
        'coverage_ratio': safe_div(coverage_count, total),
        'approx_ratio': safe_div(approx_count, total),
        'aggregated_count': aggregated_count,
        'duplicate_count': duplicate_count,
        'coverage_count': coverage_count,
        'approx_count': approx_count
    }


def render_poc_badge():
    st.markdown(f"""
        <div style='display:flex; align-items:center; justify-content:flex-start; gap:10px; padding:6px 14px; border-radius:12px; background:#0f1720; color:{INFO_TEXT};'>
            <span style='font-size:12px; color:#f5b642; font-weight:600;'>POC</span>
            <span style='font-size:12px;'>{POC_BADGE_TEXT}</span>
        </div>
    """, unsafe_allow_html=True)


def render_reliability_badge(indicators: Dict[str, float]):
    if not indicators or indicators.get('total', 0) == 0:
        return
    text = (
        f"{RELIABILITY_BADGE_TITLE}: stop_count>1 {indicators['aggregated_ratio'] * 100:.1f}% "
        f"(n={indicators['aggregated_count']}/{indicators['total']}) · "
        f"중복 {indicators['duplicate_ratio'] * 100:.1f}% (n={indicators['duplicate_count']}) · "
        f"FILE coverage {indicators['coverage_ratio'] * 100:.1f}% (n={indicators['coverage_count']}) · "
        f"시간 근사 {indicators['approx_ratio'] * 100:.1f}% (n={indicators['approx_count']})"
    )
    st.markdown(f"""
        <div style='padding:8px 12px; border-radius:14px; background:#121826; color:#cfcfcf; font-size:12px;'>
            {text}
        </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------
# Tab renderers
# ------------------------------------------------------------------

def render_process_view(views: Dict[str, pd.DataFrame], filters: Dict[str, List[str]]):
    st.subheader('Process View')
    stop_df = views['vw_stop_enriched']
    lot_df = views['vw_lot_machine_summary']
    if stop_df.empty or lot_df.empty:
        render_no_data('Process view에 적용 가능한 정지/LOT 데이터가 없습니다.')
        return
    render_sample_size_warning(len(stop_df))

    stage_summary = (
        lot_df.groupby(['stage_no'], as_index=False)
        .agg({
            'running_time_sec': 'sum',
            'power_on_time_sec': 'sum',
            'real_running_time_sec': 'sum',
            'total_stop_time_sec': 'sum',
            'stop_count': 'sum',
            'total_pickup_count': 'sum',
            'total_error_count': 'sum'
        })
    )
    stage_summary['availability'] = stage_summary.apply(
        lambda row: safe_div(row['running_time_sec'], row['power_on_time_sec']), axis=1
    )
    stage_summary['performance'] = stage_summary.apply(
        lambda row: safe_div(row['real_running_time_sec'], row['running_time_sec']), axis=1
    )
    stage_summary['quality'] = stage_summary.apply(
        lambda row: safe_div(row['total_pickup_count'] - row['total_error_count'], row['total_pickup_count']), axis=1
    )
    stage_summary['oee'] = stage_summary['availability'] * stage_summary['performance'] * stage_summary['quality']

    stop_by_stage = (
        stop_df.groupby('stage_no', as_index=False)
        .agg({'duration_sec': 'sum', 'stop_count': 'sum'})
    )
    total_stop_all = stop_by_stage['duration_sec'].sum()
    bullets: List[str] = []
    if total_stop_all and not stop_by_stage.empty:
        top_stage = stop_by_stage.loc[stop_by_stage['duration_sec'].idxmax()]
        share = safe_div(top_stage['duration_sec'], total_stop_all)
        bullets.append(
            f"관측 기반 패턴: Stage {int(top_stage['stage_no'])}이 전체 정지 시간 {format_duration(total_stop_all)}의 "
            f"{share * 100:.1f}%({format_duration(top_stage['duration_sec'])})을 차지합니다."
        )
    oee_std = stage_summary['oee'].std()
    oee_std = float(oee_std) if pd.notnull(oee_std) else 0.0
    stop_var = stage_summary['total_stop_time_sec'].var()
    stop_var = float(stop_var) if pd.notnull(stop_var) else 0.0
    stage_count = len(stage_summary)
    bullets.append(
        f"Stage 안정성: OEE 표준편차 {oee_std:.3f}(n={stage_count}) vs 정지 시간 분산 {stop_var:.1f}s²; 구조적 문제 vs 일시적 편차 가능성 관측"
    )
    total_records = len(stop_df)
    aggregated_count = int(stop_df['is_aggregated'].sum())
    event_count = total_records - aggregated_count
    aggregated_ratio = safe_div(aggregated_count, total_records)
    event_ratio = safe_div(event_count, total_records)
    macro_avg = stop_df[stop_df['is_aggregated']]['avg_stop_duration_sec'].mean()
    micro_avg = stop_df[~stop_df['is_aggregated']]['duration_sec'].mean()
    macro_avg_val = float(macro_avg) if pd.notnull(macro_avg) else 0.0
    micro_avg_val = float(micro_avg) if pd.notnull(micro_avg) else 0.0
    bullets.append(
        f"Micro vs Macro: 누적형 정지 {aggregated_count}회({aggregated_ratio * 100:.1f}%) vs 이벤트형 {event_count}회({event_ratio * 100:.1f}%), "
        f"proxy 평균 {format_duration(macro_avg_val)} vs 이벤트 평균 {format_duration(micro_avg_val)} → 건수 중심 vs 시간 중심 차이 관측"
    )
    loss_paths = []
    if total_stop_all > 0:
        path_df = (
            stop_df.groupby(['line_id', 'stage_no', 'machine_id', 'stop_reason_name'], as_index=False)
            .agg({'duration_sec': 'sum'})
            .sort_values('duration_sec', ascending=False)
            .head(3)
        )
        for _, row in path_df.iterrows():
            duration_label = format_duration(row['duration_sec'])
            share = safe_div(row['duration_sec'], total_stop_all)
            loss_paths.append(
                f"{row['line_id']} → Stage {int(row['stage_no'])} → {row['machine_id']} → {row['stop_reason_name']} "
                f"({duration_label} / {int(row['duration_sec'])}s, {share * 100:.1f}%)"
            )
        bullets.append(f"Top Loss Path: {loss_paths[0]}")
    else:
        bullets.append("Top Loss Path 계산을 위한 정지 시간이 부족합니다.")
    if not bullets:
        bullets.append('관측 기반 패턴을 만들 정지/Component 정보가 부족합니다.')
    render_insights(bullets)

    fig_stage = px.bar(
        stage_summary,
        x='stage_no',
        y='oee',
        color='oee',
        color_continuous_scale='Tealgrn',
        hover_data=['availability', 'performance', 'quality', 'total_stop_time_sec', 'stop_count'],
        title='Stage OEE',
        template=DARK_TEMPLATE
    )
    fig_stage.update_traces(hovertemplate='Stage %{x}<br>OEE %{y:.2f}<br>Av %{customdata[0]:.2f}<br>Perf %{customdata[1]:.2f}<br>Qual %{customdata[2]:.2f}<extra></extra>')
    st.plotly_chart(fig_stage, use_container_width=True)

    fig_pie = px.pie(
        stop_by_stage,
        names='stage_no',
        values='duration_sec',
        title='Stage Stop Time Contribution',
        template=DARK_TEMPLATE
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    if not stop_df.empty:
        reason_tot = (
            stop_df.groupby(['stop_reason_code', 'stop_reason_name'], as_index=False)['duration_sec']
            .sum()
            .sort_values('duration_sec', ascending=False)
        )
        reason_tot['cum_pct'] = reason_tot['duration_sec'].cumsum() / (reason_tot['duration_sec'].sum() or 1)
        pareto = go.Figure()
        pareto.add_trace(go.Bar(
            x=reason_tot['stop_reason_name'],
            y=reason_tot['duration_sec'],
            name='Stop Time',
            marker_color=PRIMARY_COLOR
        ))
        pareto.add_trace(go.Scatter(
            x=reason_tot['stop_reason_name'],
            y=reason_tot['cum_pct'] * 100,
            name='Cumulative %',
            mode='lines+markers',
            marker_color=SECONDARY_COLOR,
            yaxis='y2'
        ))
        pareto.update_layout(
            template=DARK_TEMPLATE,
            title='Stop Pareto (누적 80% 기준)',
            yaxis_title='Stop Time (s)',
            yaxis2=dict(title='Cumulative %', overlaying='y', side='right', range=[0, 110])
        )
        pareto.add_shape(
            type='line',
            x0=-0.5,
            x1=len(reason_tot) - 0.5,
            y0=80,
            y1=80,
            yref='y2',
            line=dict(color='#ffffff', width=1, dash='dash')
        )
        st.plotly_chart(pareto, use_container_width=True)
        top80_idx = reason_tot[reason_tot['cum_pct'] >= 0.8]
        if not top80_idx.empty:
            idx = top80_idx.index[0]
            covered = reason_tot.loc[:idx, 'duration_sec'].sum()
            n_reasons = idx + 1
        else:
            covered = reason_tot['duration_sec'].sum()
            n_reasons = len(reason_tot)
        insight_text = (
            f"상위 {n_reasons}개 정지 원인이 전체 {format_duration(reason_tot['duration_sec'].sum())} 중 "
            f"{format_duration(covered)} ({safe_div(covered, reason_tot['duration_sec'].sum()) * 100:.1f}%)을 차지합니다."
        )
        st.caption(insight_text)

    approx_note = highlight_time_note(stop_df['time_axis_is_approx'].any())
    trend = (
        stop_df.dropna(subset=['time_sort'])
        .groupby('time_sort', as_index=False)
        .agg({'stop_count': 'sum', 'duration_sec': 'sum'})
        .sort_values('time_sort')
    )
    if trend.empty:
        render_no_data('시간 축을 만들 수 없어 트렌드를 표시할 수 없습니다.')
    else:
        fig_line = px.line(
            trend,
            x='time_sort',
            y='duration_sec',
            markers=True,
            title=f'Stop Time Trend{approx_note}',
            template=DARK_TEMPLATE
        )
        fig_line.update_traces(hovertemplate='Time %{x}<br>Stop Time %{y:.0f}s<extra></extra>')
        st.plotly_chart(fig_line, use_container_width=True)

    lot_stats = (
        lot_df.groupby(['lot_id', 'lot_name'], as_index=False)
        .agg({
            'total_stop_time_sec': 'sum',
            'stop_count': 'sum',
            'total_pickup_count': 'sum',
            'total_error_count': 'sum',
            'oee': 'mean'
        })
    )
    lot_reason = (
        stop_df.groupby(['lot_id', 'stop_reason_name'], as_index=False)['duration_sec']
        .sum()
    )
    lot_reason = lot_reason.sort_values(['lot_id', 'duration_sec'], ascending=[True, False])
    lot_reason = lot_reason.groupby('lot_id').first().reset_index()
    lot_summary = lot_stats.merge(lot_reason[['lot_id', 'stop_reason_name']], on='lot_id', how='left')
    lot_summary['error_rate'] = lot_summary.apply(
        lambda row: safe_div(row['total_error_count'], row['total_pickup_count']), axis=1
    )
    lot_summary['stop_time'] = lot_summary['total_stop_time_sec'].apply(format_duration)
    lot_summary = lot_summary.sort_values('total_stop_time_sec', ascending=False)
    st.dataframe(
        lot_summary[['lot_name', 'stop_time', 'stop_count', 'error_rate', 'oee', 'stop_reason_name']]
        .assign(error_rate=lambda df: df['error_rate'].apply(lambda v: f"{v * 100:.1f}%"))
        .rename(columns={'stop_reason_name': 'Top Stop Reason'})
    )


def render_machine_view(views: Dict[str, pd.DataFrame], filters: Dict[str, List[str]]):
    st.subheader('Machine View')
    machine_df = views['vw_lot_machine_summary']
    stop_df = views['vw_stop_enriched']
    component_df = views['vw_component_quality']
    if machine_df.empty:
        render_no_data('Machine view에 적용 가능한 summary 데이터가 없습니다.')
        return
    render_sample_size_warning(len(stop_df))
    render_sample_size_warning(len(component_df))

    bullets: List[str] = []
    total_stop_time = stop_df['duration_sec'].sum()
    if not stop_df.empty and total_stop_time:
        top_machine = stop_df.groupby('machine_id')['duration_sec'].sum().sort_values(ascending=False).reset_index()
        if not top_machine.empty:
            row = top_machine.iloc[0]
            share = safe_div(row['duration_sec'], total_stop_time)
            bullets.append(
                f"관측 기반 패턴: Machine {row['machine_id']}이 정지 시간 {format_duration(row['duration_sec'])} "
                f"({share * 100:.1f}% of {format_duration(total_stop_time)})을 차지합니다."
            )

    changeover_machines = stop_df[stop_df['stop_reason_code'] == 'CHANGEOVER']['machine_id'].unique()
    overall_error_rate = safe_div(component_df['error_count'].sum(), component_df['pickup_count'].sum())
    if len(changeover_machines) > 0 and not component_df.empty:
        subset = component_df[component_df['machine_id'].isin(changeover_machines)]
        changeover_error_rate = safe_div(subset['error_count'].sum(), subset['pickup_count'].sum())
        bullets.append(
            f"동반 관측 가능성: Changeover 관련 Machine error_rate {changeover_error_rate * 100:.1f}% "
            f"(n={int(subset['pickup_count'].sum())}) vs 전체 {overall_error_rate * 100:.1f}%."
        )

    running_totals = machine_df.groupby('machine_id')['running_time_sec'].sum()
    if not running_totals.empty:
        threshold = running_totals.quantile(0.75)
        high_running = running_totals[running_totals >= threshold].index.tolist()
        high_error_df = component_df[component_df['machine_id'].isin(high_running)]
        high_error_rate = safe_div(high_error_df['error_count'].sum(), high_error_df['pickup_count'].sum())
        bullets.append(
            f"과부하 상태 가능성: 상위 Running 시간 Machine error_rate {high_error_rate * 100:.1f}% "
            f"vs 전체 {overall_error_rate * 100:.1f}% (상위 {len(high_running)}대)."
        )

    render_insights(bullets)

    breakdown = (
        machine_df.groupby('machine_id', as_index=False)[['running_time_sec', 'total_stop_time_sec', 'transfer_time_sec', 'placement_time_sec', 'board_recognition_time_sec']]
        .sum()
    )
    melt = breakdown.melt(id_vars='machine_id', var_name='segment', value_name='seconds')
    fig_stack = px.bar(
        melt,
        x='machine_id',
        y='seconds',
        color='segment',
        title='Machine Time Breakdown',
        template=DARK_TEMPLATE
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    reason_tally = (
        stop_df.groupby(['stop_reason_code', 'stop_reason_name', 'stop_reason_group'], as_index=False)
        .agg({'duration_sec': 'sum', 'stop_count': 'sum'})
    )
    chart_metric = st.radio('Stop reason top10 기준', ['duration_sec', 'stop_count'], horizontal=True)
    metric_label = 'Stop Time' if chart_metric == 'duration_sec' else 'Stop Count'
    reason_display = reason_tally.sort_values(chart_metric, ascending=False).head(10)
    fig_reason = px.bar(
        reason_display,
        x='stop_reason_name',
        y=chart_metric,
        color='stop_reason_group',
        title=f'Stop Reason Top10 ({metric_label})',
        template=DARK_TEMPLATE,
        color_discrete_map=STOP_REASON_GROUP_COLORS,
        hover_data=['stop_count', 'duration_sec']
    )
    st.plotly_chart(fig_reason, use_container_width=True)

    if not reason_tally.empty:
        total_time = reason_tally['duration_sec'].sum()
        quadrant = reason_tally.copy()
        quadrant['share'] = quadrant['duration_sec'] / (total_time or 1)
        quadrant['duration_label'] = quadrant['duration_sec'].apply(format_duration)
        quad_fig = px.scatter(
            quadrant,
            x='stop_count',
            y='duration_sec',
            size='duration_sec',
            color='stop_reason_group',
            template=DARK_TEMPLATE,
            title='Stop Reason Time-vs-Count Quadrant',
            color_discrete_map=STOP_REASON_GROUP_COLORS,
            text='stop_reason_name',
            custom_data=['duration_label', 'share']
        )
        quad_fig.update_traces(
            hovertemplate='<b>%{text}</b><br>Stop Time %{customdata[0]}<br>Stop Count %{x}<br>Share %{customdata[1]:.1%}<extra></extra>'
        )
        st.plotly_chart(quad_fig, use_container_width=True)

        time_top = quadrant.loc[quadrant['duration_sec'].idxmax()]
        count_top = quadrant.loc[quadrant['stop_count'].idxmax()]
        quad_insights = [
            f"관측 기반 패턴: {time_top['stop_reason_name']}이 정지 시간 {format_duration(time_top['duration_sec'])} ({time_top['share'] * 100:.1f}% share) 으로 가장 높습니다.",
            f"관측 기반 패턴: {count_top['stop_reason_name']}이 건수 {int(count_top['stop_count'])}회로 상위입니다."
        ]

        st.markdown("<br>".join(quad_insights))

    machine_trend = (
        machine_df.groupby(['machine_id', 'day'], as_index=False)
        .agg({
            'running_time_sec': 'sum',
            'real_running_time_sec': 'sum',
            'power_on_time_sec': 'sum',
            'total_pickup_count': 'sum',
            'total_error_count': 'sum'
        })
    )
    machine_trend['availability'] = machine_trend.apply(
        lambda row: safe_div(row['running_time_sec'], row['power_on_time_sec']), axis=1
    )
    machine_trend['performance'] = machine_trend.apply(
        lambda row: safe_div(row['real_running_time_sec'], row['running_time_sec']), axis=1
    )
    machine_trend['quality'] = machine_trend.apply(
        lambda row: safe_div(row['total_pickup_count'] - row['total_error_count'], row['total_pickup_count']), axis=1
    )
    machine_trend['oee'] = machine_trend['availability'] * machine_trend['performance'] * machine_trend['quality']
    machine_focus = st.multiselect('Machine Trend Focus', sorted(machine_trend['machine_id'].unique()), default=sorted(machine_trend['machine_id'].unique())[:3])
    if not machine_focus:
        machine_focus = sorted(machine_trend['machine_id'].unique())[:1]
    trend_target = machine_trend[machine_trend['machine_id'].isin(machine_focus)]
    if trend_target.empty:
        render_no_data('선택된 Machine에 대한 트렌드 데이터가 없습니다.')
    else:
        fig_oee = px.line(
            trend_target,
            x='day',
            y='oee',
            color='machine_id',
            markers=True,
            template=DARK_TEMPLATE,
            title='Machine OEE Trend'
        )
        fig_oee.update_traces(hovertemplate='Machine %{legendgroup}<br>Day %{x}<br>OOE %{y:.2f}<extra></extra>')
        st.plotly_chart(fig_oee, use_container_width=True)

    if not component_df.empty:
        component_df['error_rate'] = component_df['error_rate'].replace(np.nan, 0)
        top_components = (
            component_df
            .sort_values('error_rate', ascending=False)
            .head(10)
            [['part_number', 'feeder_id', 'nozzle_serial', 'machine_id', 'error_rate', 'error_count']]
        )
        top_components['error_rate'] = top_components['error_rate'].apply(lambda v: f"{v * 100:.1f}%")
        st.table(top_components)
    else:
        render_no_data('Component defect data가 없습니다.')

    # ------------------------------------------------------------------
    # Pickup RCA Analysis
    # ------------------------------------------------------------------
    min_pickup = filters.get('min_pickup_count', 0)
    pickup_quality = component_df[component_df['pickup_count'] >= min_pickup] if not component_df.empty else component_df
    if pickup_quality.empty:
        st.info("Pickup RCA Analysis를 위한 충분한 data가 없습니다.")
        return

    pickup_quality = pickup_quality.copy()
    pickup_quality['pickup_error_rate'] = pickup_quality.apply(
        lambda row: safe_div(row['pickup_error_count'], row['pickup_count']), axis=1
    )

    st.markdown("## Pickup RCA Analysis")
    machine_pickup_all = (
        pickup_quality.groupby('machine_id', as_index=False)
        .agg(pickup_error_count=('pickup_error_count', 'sum'), pickup_count=('pickup_count', 'sum'))
    )
    machine_pickup_all['rate'] = machine_pickup_all.apply(
        lambda row: safe_div(row['pickup_error_count'], row['pickup_count']), axis=1
    )
    machine_pickup = machine_pickup_all.sort_values('rate', ascending=False).head(10)
    overall_rate = safe_div(machine_pickup_all['pickup_error_count'].sum(), machine_pickup_all['pickup_count'].sum())

    bar_fig = px.bar(
        machine_pickup[::-1],
        x='rate',
        y='machine_id',
        orientation='h',
        color='rate',
        color_continuous_scale='Tealgrn',
        template=DARK_TEMPLATE,
        title='Pickup Error Machine Ranking',
        custom_data=['pickup_error_count', 'pickup_count']
    )
    bar_fig.update_traces(
        hovertemplate='<b>%{y}</b><br>Pickup Error Rate %{x:.1%}<br>Error %{customdata[0]} / Pickup %{customdata[1]}<extra></extra>'
    )
    st.plotly_chart(bar_fig, use_container_width=True)
    st.table(
        machine_pickup[['machine_id', 'pickup_error_count', 'pickup_count', 'rate']]
        .rename(columns={'machine_id': 'Machine', 'pickup_error_count': 'Error Count', 'pickup_count': 'Pickup Count', 'rate': 'Error Rate'})
    )
    if not machine_pickup.empty:
        top_machine = machine_pickup.iloc[0]
        multiplier = safe_div(top_machine['rate'], overall_rate)
        st.markdown(
            f"관측 기반 패턴: Machine {top_machine['machine_id']}의 Pickup Error Rate가 "
            f"{top_machine['rate'] * 100:.1f}% ({int(top_machine['pickup_error_count'])}/{int(top_machine['pickup_count'])})로 "
            f"평균 대비 {multiplier:.1f}배 높게 관측됩니다."
        )

    col1, col2 = st.columns(2)

    with col1:
        corr_df = machine_pickup_all[['machine_id', 'pickup_error_count', 'pickup_count', 'rate']].copy()
        stops_per_machine = stop_df.groupby('machine_id', as_index=False)['duration_sec'].sum()
        total_stop = stops_per_machine['duration_sec'].sum()
        stops_per_machine['stop_share'] = stops_per_machine['duration_sec'] / (total_stop or 1)
        corr_df = corr_df.merge(stops_per_machine[['machine_id', 'stop_share']], on='machine_id', how='left').fillna(0)
        corr_df['pickup_count'] = corr_df['pickup_count'].fillna(0)
        stop_group = (
            stop_df.groupby(['machine_id', 'stop_reason_group'], as_index=False)['duration_sec']
            .sum()
        )
        #stop_group = stop_group.loc[stop_group.groupby('machine_id')['duration_sec'].idxmax().fillna(stop_group.index)]
        idx = stop_group.groupby('machine_id')['duration_sec'].idxmax()
        # NaN 제거 후 선택
        idx = idx.dropna()
        stop_group = stop_group.loc[idx]
        
        corr_df = corr_df.merge(stop_group[['machine_id', 'stop_reason_group']], on='machine_id', how='left')
        scatter = px.scatter(
            corr_df,
            x='rate',
            y='stop_share',
            size='pickup_count',
            color='stop_reason_group',
            template=DARK_TEMPLATE,
            title='Pickup vs Stop Correlation',
            hover_data={'pickup_count': True, 'stop_share': ':.1%', 'rate': ':.1%'}
        )
        mean_rate = corr_df['rate'].mean()
        mean_stop = corr_df['stop_share'].mean()
        scatter.add_shape({'type': 'line', 'x0': mean_rate, 'x1': mean_rate, 'y0': 0, 'y1': 1, 'line': {'dash': 'dash', 'color': '#ffffff'}})
        scatter.add_shape({'type': 'line', 'x0': 0, 'x1': 1, 'y0': mean_stop, 'y1': mean_stop, 'line': {'dash': 'dash', 'color': '#ffffff'}})
        st.plotly_chart(scatter, use_container_width=True)
        corr_coef = corr_df['rate'].corr(corr_df['stop_share'])
        if corr_coef and corr_coef > 0.3:
            st.markdown("관측 기반 패턴: Pickup Error가 높은 설비에서 정지 비중도 동반 상승하는 경향이 보입니다.")

        st.markdown("""
        **Pickup ↔ Stop Correlation 코멘트**
        - **Equipment**: Pickup Error Rate↑, Stop Share↑ → 설비 조건 불안정 가능성 / Vacuum·헤드·노즐 점검 필요
        - **Quality**: Pickup Error Rate↑, Stop Share↑ → 부품·라이브러리·릴 집중 여부 추가 확인 필요
        - **Waiting**: Pickup Error Rate 낮은데 Stop Share↑ → 라인 밸런스 문제 (Pickup 직접 연계 가능성 낮음)
        - **Process**: CHANGEOVER 或 특정 Lot 중심 증가 → 운영 조건 이슈 가능성
        """)

    with col2:
        perf_df = (
            machine_df.groupby('machine_id', as_index=False)
            .agg(real_running_time_sec=('real_running_time_sec', 'sum'), running_time_sec=('running_time_sec', 'sum'))
        )
        perf_df['performance'] = perf_df.apply(
            lambda row: safe_div(row['real_running_time_sec'], row['running_time_sec']), axis=1
        )
        perf_df = perf_df.merge(machine_pickup[['machine_id', 'rate', 'pickup_count']], on='machine_id', how='left').fillna(0)
        perf_df = perf_df.sort_values('rate', ascending=False).head(10)
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Bar(
            x=perf_df['machine_id'],
            y=perf_df['rate'],
            name='Pickup Error Rate',
            marker_color=PRIMARY_COLOR,
            customdata=perf_df[['pickup_count']],
            hovertemplate='Machine %{x}<br>Pickup Error %{y:.1%}<br>Pickup Count %{customdata[0]}<extra></extra>'
        ))
        fig_perf.add_trace(go.Scatter(
            x=perf_df['machine_id'],
            y=perf_df['performance'],
            name='Performance',
            yaxis='y2',
            marker_color=SECONDARY_COLOR,
            hovertemplate='Machine %{x}<br>Performance %{y:.1%}<extra></extra>'
        ))
        fig_perf.update_layout(
            template=DARK_TEMPLATE,
            title='Pickup Error vs Performance',
            yaxis=dict(title='Pickup Error Rate', tickformat=',.0%'),
            yaxis2=dict(title='Performance', overlaying='y', side='right')
        )
        st.plotly_chart(fig_perf, use_container_width=True)
        low_perf = perf_df.loc[perf_df['performance'].idxmin()]
        st.markdown(
            f"관측 기반 패턴: Machine {low_perf['machine_id']}은 Pickup Error {low_perf['rate'] * 100:.1f}% "
            f"대비 Performance {low_perf['performance'] * 100:.1f}%으로 운영 조건 저하 가능성이 있습니다."
        )

    lot_pickup = (
        pickup_quality.groupby('lot_id', as_index=False)
        .agg(pickup_error_count=('pickup_error_count', 'sum'), pickup_count=('pickup_count', 'sum'))
    )
    if lot_pickup.empty:
        st.info("Lot Concentration 분석을 위한 데이터가 없습니다.")
        return
    lot_pickup['rate'] = lot_pickup.apply(
        lambda row: safe_div(row['pickup_error_count'], row['pickup_count']), axis=1
    )
    lot_pickup = lot_pickup.sort_values('rate', ascending=False)
    lot_fig = px.bar(
        lot_pickup,
        x='lot_id',
        y='rate',
        template=DARK_TEMPLATE,
        title='Lot Concentration Analysis',
        hover_data={'pickup_error_count': True, 'pickup_count': True}
    )
    lot_fig.update_traces(hovertemplate='Lot %{x}<br>Pickup Error %{y:.1%}<br>Error %{customdata[0]} / Pickup %{customdata[1]}<extra></extra>')
    st.plotly_chart(lot_fig, use_container_width=True)
    total_lot_error = lot_pickup['pickup_error_count'].sum()
    top_share = safe_div(lot_pickup.iloc[0]['pickup_error_count'], total_lot_error)
    if top_share > 0.4:
        st.markdown("특정 LOT에 오류가 집중되는 경향이 관측됩니다.")


def render_quality_view(views: Dict[str, pd.DataFrame], filters: Dict[str, List[str]]):
    st.subheader('Quality View')
    component_df = views['vw_component_quality']
    stop_df = views['vw_stop_enriched']
    if component_df.empty:
        render_no_data('Quality view를 위한 component data가 부족합니다.')
        return
    render_sample_size_warning(len(component_df))
    render_sample_size_warning(len(stop_df))

    bullets: List[str] = []
    total_errors = component_df['error_count'].sum()
    defect_agg = (
        component_df.groupby('defect_type', as_index=False)['error_count']
        .sum()
    )
    if not defect_agg.empty:
        top_defect = defect_agg.loc[defect_agg['error_count'].idxmax()]
        share = safe_div(top_defect['error_count'], total_errors)
        bullets.append(
            f"동반 관측 가능성: {top_defect['defect_type']}이 전체 오류 {int(total_errors)}건 중 "
            f"{share * 100:.1f}%({int(top_defect['error_count'])}건)을 차지합니다."
        )

    part_lot = (
        component_df.groupby(['part_number', 'lot_id'], as_index=False)
        .agg({'error_count': 'sum', 'pickup_count': 'sum'})
    )
    part_lot['error_rate'] = part_lot.apply(lambda row: safe_div(row['error_count'], row['pickup_count']), axis=1)
    part_variance = (
        part_lot.groupby('part_number')['error_rate']
        .var()
        .reset_index(name='variance')
        .dropna()
    )
    if not part_variance.empty:
        top_part = part_variance.sort_values('variance', ascending=False).iloc[0]
        top_lot_row = part_lot[part_lot['part_number'] == top_part['part_number']].sort_values('error_rate', ascending=False).head(1)
        lot_label = top_lot_row.iloc[0]['lot_id'] if not top_lot_row.empty else 'Unknown'
        bullets.append(
            f"Part Hotspot 안정성: {top_part['part_number']} LOT {lot_label} error_rate 분산 {top_part['variance']:.4f}; 특정 LOT 집중 현상 여부 관측"
        )

    machine_rate_summary = (
        component_df.groupby('machine_id', as_index=False)
        .agg({'error_count': 'sum', 'pickup_count': 'sum'})
    )
    machine_rate_summary['error_rate'] = machine_rate_summary.apply(
        lambda row: safe_div(row['error_count'], row['pickup_count']), axis=1
    )
    if not machine_rate_summary.empty:
        top_machine = machine_rate_summary.sort_values('error_rate', ascending=False).iloc[0]
        bullets.append(
            f"Machine {top_machine['machine_id']} error_rate {top_machine['error_rate'] * 100:.1f}% "
            f"({int(top_machine['error_count'])} / {int(top_machine['pickup_count'])}); 동반 관측 가능성 확인 필요"
        )

    if not bullets:
        bullets.append('관측 기반 패턴을 만들 품질/정지 정보가 부족합니다.')
    render_insights(bullets)

    fig_pie = px.pie(defect_agg, names='defect_type', values='error_count', template=DARK_TEMPLATE, title='Error Type Distribution')
    st.plotly_chart(fig_pie, use_container_width=True)

    top_parts = (
        component_df.groupby('part_number', as_index=False)
        .agg({'error_count': 'sum', 'pickup_count': 'sum'})
    )
    top_parts['error_rate'] = top_parts.apply(
        lambda row: safe_div(row['error_count'], row['pickup_count']), axis=1
    )
    metric_toggle = st.radio('Heatmap Metric', ['Error Count', 'Error Rate'], horizontal=True)
    heat_metric = 'error_count' if metric_toggle == 'Error Count' else 'error_rate'
    heat_base = component_df.groupby(['machine_id', 'part_number'], as_index=False).agg({'error_count': 'sum', 'pickup_count': 'sum'})
    if heat_metric == 'Error Rate':
        heat_base['metric'] = heat_base.apply(lambda row: safe_div(row['error_count'], row['pickup_count']), axis=1)
    else:
        heat_base['metric'] = heat_base['error_count']
    top_parts_list = heat_base.sort_values('metric', ascending=False).head(10)
    heatmap = top_parts_list.pivot(index='machine_id', columns='part_number', values='metric').fillna(0)
    fig_heat = go.Figure(
        data=go.Heatmap(
            z=heatmap.values,
            x=heatmap.columns,
            y=heatmap.index,
            colorscale='Viridis',
            hovertemplate='Machine %{y}<br>Part %{x}<br>' + ('Error Rate %{z:.1%}' if heat_metric == 'error_rate' else 'Error Count %{z}<extra></extra>')
        )
    )
    fig_heat.update_layout(title=f'Component Defect Heatmap ({metric_toggle})', template=DARK_TEMPLATE)
    st.plotly_chart(fig_heat, use_container_width=True)

    assoc_machine = None
    filter_machines = filters.get('machines', [])
    machine_stops = stop_df.groupby('machine_id')['duration_sec'].sum()
    if filter_machines:
        assoc_machine = filter_machines[0]
    elif not machine_stops.empty:
        assoc_machine = machine_stops.idxmax()

    overall_errors = component_df['error_count'].sum()
    overall_pickups = component_df['pickup_count'].sum()
    overall_rate = safe_div(overall_errors, overall_pickups)

    if assoc_machine:
        machine_scope = component_df[component_df['machine_id'] == assoc_machine]
        machine_errors = machine_scope['error_count'].sum()
        machine_pickups = machine_scope['pickup_count'].sum()
        machine_rate = safe_div(machine_errors, machine_pickups)
        ratio_text = (
            f"평균 대비 {safe_div(machine_rate, overall_rate):.1f}배 높게 관측됩니다."
            if overall_rate else "전체 기준이 부족하여 평균 대비 비교 불가합니다."
        )
        st.markdown(
            f"Stop ↔ Quality 동반 관측 가능성: Machine {assoc_machine} error_rate {machine_rate * 100:.1f}% "
            f"({int(machine_errors)}/{int(machine_pickups)}) vs 전체 {overall_rate * 100:.1f}% "
            f"({int(overall_errors)}/{int(overall_pickups)}) · {ratio_text}"
        )
    else:
        st.info('Stop 로그 대상 Machine을 선택하거나 데이터 확보 후 실행해주세요.')

    changeover_machines = stop_df[stop_df['stop_reason_code'] == 'CHANGEOVER']['machine_id'].unique()
    if len(changeover_machines) and overall_rate:
        changeover_scope = component_df[component_df['machine_id'].isin(changeover_machines)]
        if not changeover_scope.empty:
            co_errors = changeover_scope['error_count'].sum()
            co_pickups = changeover_scope['pickup_count'].sum()
            co_rate = safe_div(co_errors, co_pickups)
            st.markdown(
                f"Changeover 동반 관측 가능성: 관련 Machine error_rate {co_rate * 100:.1f}% "
                f"({int(co_errors)}/{int(co_pickups)}) vs 평균 {overall_rate * 100:.1f}% "
                f"({int(overall_errors)}/{int(overall_pickups)}) → 평균 대비 {safe_div(co_rate, overall_rate):.1f}배 높게 관측됩니다."
            )

    trend_base = stop_df.dropna(subset=['time_sort']).copy()
    if 'error_count' not in trend_base.columns:
        trend_base['error_count'] = 1
    error_trend = (
        trend_base
            .groupby('time_sort', as_index=False)
            .agg(error_count=('error_count', 'sum'))
    )
    approx_note = highlight_time_note(stop_df['time_axis_is_approx'].any())
    if not error_trend.empty:
        fig_error_line = px.line(
            error_trend,
            x='time_sort',
            y='error_count',
            markers=True,
            template=DARK_TEMPLATE,
            title=f'Error Trend{approx_note}'
        )
        st.plotly_chart(fig_error_line, use_container_width=True)

    machine_rate_summary['top_part'] = (
        component_df.sort_values('error_count', ascending=False)
        .drop_duplicates('machine_id')
        .set_index('machine_id')['part_number']
    )
    machine_rate_summary['top_part'] = machine_rate_summary['top_part'].fillna('-')
    machine_rate_summary['error_rate'] = machine_rate_summary['error_rate'].apply(lambda v: f"{v * 100:.1f}%")
    if not machine_rate_summary.empty:
        st.dataframe(machine_rate_summary[['machine_id', 'error_rate', 'error_count', 'pickup_count', 'top_part']])
# ------------------------------------------------------------------
# RCA summary
# ------------------------------------------------------------------
@st.cache_data(ttl=300)
def generate_rca_summary(views: Dict[str, pd.DataFrame], filters: Dict[str, List[str]]) -> Tuple[str, str]:
    stop_df = views['vw_stop_enriched']
    machine_df = views['vw_lot_machine_summary']
    component_df = views['vw_component_quality']

    scope = entropy_insight_label(filters)
    scope += ' | ' + summarize_filters(filters)

    total_stop = stop_df['duration_sec'].sum() if not stop_df.empty else 0
    loss_paths: List[str] = []
    if total_stop > 0:
        path_df = (
            stop_df.groupby(['line_id', 'stage_no', 'machine_id', 'stop_reason_name'], as_index=False)
            .agg({'duration_sec': 'sum'})
            .sort_values('duration_sec', ascending=False)
            .head(3)
        )
        for _, row in path_df.iterrows():
            duration_label = format_duration(row['duration_sec'])
            share = safe_div(row['duration_sec'], total_stop)
            loss_paths.append(
                f"{row['line_id']} → Stage {int(row['stage_no'])} → {row['machine_id']} → {row['stop_reason_name']} "
                f"({duration_label} / {int(row['duration_sec'])}s, {share * 100:.1f}%)"
            )
    else:
        loss_paths.append('정지 데이터가 부족하여 Loss Path를 계산할 수 없습니다.')

    process_finding = 'Process: 데이터 부족'
    quality_finding = 'Quality: 데이터 부족'
    equipment_finding = 'Equipment: 데이터 부족'
    if not machine_df.empty:
        stage_stats = machine_df.groupby('stage_no')['total_stop_time_sec'].sum().reset_index()
        if not stage_stats.empty:
            top_stage = stage_stats.loc[stage_stats['total_stop_time_sec'].idxmax()]
            share = safe_div(top_stage['total_stop_time_sec'], stage_stats['total_stop_time_sec'].sum())
            process_finding = (
                f"Stage {int(top_stage['stage_no'])} 정지가 {format_duration(top_stage['total_stop_time_sec'])} "
                f"({share * 100:.1f}%); 구조적 vs 편차 가능성 관측 기반 패턴"
            )
    if not component_df.empty:
        part_stats = component_df.groupby('part_number')['error_count'].sum().reset_index()
        if not part_stats.empty:
            top_part = part_stats.loc[part_stats['error_count'].idxmax()]
            quality_finding = (
                f"{top_part['part_number']}이 {int(top_part['error_count'])}개 오류; LOT 편중 가능성 관측 기반"
            )
        machine_errors = component_df.groupby('machine_id')['error_count'].sum().reset_index()
        if not machine_errors.empty:
            top_machine = machine_errors.loc[machine_errors['error_count'].idxmax()]
            equipment_finding = (
                f"Machine {top_machine['machine_id']} 오류 {int(top_machine['error_count'])}건; 설비 집중 점검 필요 가능성"
            )

    exec_lines: List[str] = [
        f"Executive Summary ({scope})",
        *[f"- Top Loss Path {idx + 1}: {path}" for idx, path in enumerate(loss_paths)],
        f"- 핵심 손실 구조: {process_finding}",
        f"- 핵심 관측 패턴: {quality_finding}",
        f"- 주요 조치 방향: {equipment_finding}"
        #"- 제한사항: 관측 기반, 시간축 근사(파일 시퀀스), 누적 proxy 여부 확인 필요"
    ]

    dominant_reason = '정보 부족'
    dominant_share = 0
    if not stop_df.empty and total_stop > 0:
        reason_stats = (
            stop_df.groupby('stop_reason_name')['duration_sec']
            .sum()
            .reset_index()
            .sort_values('duration_sec', ascending=False)
        )
        if not reason_stats.empty:
            top_reason = reason_stats.iloc[0]
            dominant_reason = top_reason['stop_reason_name']
            dominant_share = safe_div(top_reason['duration_sec'], total_stop)

    stop_counts_total = stop_df['stop_count'].sum()
    avg_stop_time = safe_div(total_stop, stop_counts_total)
    avg_stop_label = format_duration(avg_stop_time)

    focus_machines = filters.get('machines') or []
    if not focus_machines and not stop_df.empty:
        focus_machines = stop_df.groupby('machine_id')['duration_sec'].sum().nlargest(2).index.tolist()
    target_machine_line = ', '.join(focus_machines) if focus_machines else '전체'

    hot_part_info = '정보 부족'
    if not component_df.empty:
        top_component = component_df.sort_values('error_count', ascending=False).head(1)
        if not top_component.empty:
            row = top_component.iloc[0]
            hot_part_info = (
                f"{row['part_number']} / Feeder {row.get('feeder_id','-')} / Nozzle {row.get('nozzle_serial','-')} "
                f"({int(row['error_count'])}건 오류)"
            )

    action_codes = extract_top_stop_reasons(stop_df)
    check_points: List[str] = []
    for code in action_codes:
        for template in ACTION_TEMPLATES.get(code, []):
            check_points.append(f"[{code}] {template}")
            if len(check_points) >= 5:
                break
        if len(check_points) >= 5:
            break
    while len(check_points) < 5:
        check_points.append('관측 기반 추가 분석을 통해 우선순위 Action을 선정하세요.')

    operator_lines: List[str] = [
        "Operator Summary",
        f"- 대상 설비: {target_machine_line}",
        f"- 주요 정지 유형: {dominant_reason} ({dominant_share * 100:.1f}%, {int(total_stop)}s)",
        f"- 평균 정지시간: {avg_stop_label} (총 {int(total_stop)}s / {int(stop_counts_total)}회)",
        f"- Hot part / feeder / nozzle: {hot_part_info}",
        "- 점검 포인트:",
        *[f"  {idx + 1}. {point}" for idx, point in enumerate(check_points)]
        #"- 데이터 한계: 관측 기반, 시간축 근사(파일 시퀀스), 누적 proxy 여부 확인 필요"
    ]

    return '\n'.join(exec_lines), '\n'.join(operator_lines)

def render_rca_summary(views: Dict[str, pd.DataFrame], filters: Dict[str, List[str]]):
    st.subheader("RCA Summary")
   
    exec_summary, operator_summary = generate_rca_summary(views, filters)
    stop_df = views['vw_stop_enriched']
    total_time = stop_df['duration_sec'].sum()
    top_paths = (
        stop_df.groupby(['line_id', 'stage_no', 'machine_id', 'stop_reason_name'], as_index=False)
        .agg(stop_time=('duration_sec', 'sum'), stop_count=('stop_count', 'sum'))
        .sort_values('stop_time', ascending=False)
        .head(3)
    )

    st.markdown("### Executive Summary")
    if top_paths.empty:
        st.info("Top Loss Path 데이터를 확보할 수 없습니다.")
    else:
        cols = st.columns(len(top_paths))
        for col, (_, row) in zip(cols, top_paths.iterrows()):
            share = safe_div(row['stop_time'], total_time)
            route = f"{row['line_id']} → Stage {int(row['stage_no'])} → {row['machine_id']} → {row['stop_reason_name']}"
            with col:
                st.markdown(f"**{route}**")
                st.markdown(f"Stop Time: {_hms(row['stop_time'])} / {int(row['stop_time'])}s")
                st.markdown(f"Stop Count: {int(row['stop_count'])}")
                st.markdown(f"Share: {share * 100:.1f}%")
                st.progress(min(max(share, 0.0), 1.0))

    st.text_area("Copy", exec_summary, height=220)

    st.markdown("### Operator Summary")
    component_df = views['vw_component_quality']
    min_pickup = filters.get('min_pickup_count', 0)
    comp_filtered = component_df[component_df['pickup_count'] >= min_pickup]
    col_left, col_right = st.columns(2)
    if comp_filtered.empty:
        col_left.info("필터 조건을 만족하는 Machine data가 없습니다.")
        col_right.info("필터 조건을 만족하는 Part data가 없습니다.")
    else:
        machine_stats = (
            comp_filtered.groupby('machine_id', as_index=False)
            .agg(error_count=('error_count', 'sum'), pickup_count=('pickup_count', 'sum'))
        )
        machine_stats['error_rate'] = machine_stats.apply(
            lambda row: safe_div(row['error_count'], row['pickup_count']), axis=1
        )
        top_parts_machine = (
            comp_filtered.sort_values('error_count', ascending=False)
            .drop_duplicates('machine_id')[['machine_id', 'part_number']]
        )
        machines_view = (
            machine_stats.merge(top_parts_machine, on='machine_id', how='left')
            .sort_values('error_rate', ascending=False)
            .head(10)
        )
        machines_view['error_rate'] = machines_view.apply(
            lambda row: f"{row['error_rate'] * 100:.1f}% ({int(row['error_count'])}/{int(row['pickup_count'])})", axis=1
        )
        machines_view = machines_view[['machine_id', 'error_rate', 'part_number']]
        machines_view = machines_view.rename(columns={'machine_id': 'Machine', 'part_number': 'Top Part'})
        col_left.markdown("#### Worst Machines (Top 10)")
        col_left.table(machines_view)

        part_stats = (
            comp_filtered.groupby('part_number', as_index=False)
            .agg(error_count=('error_count', 'sum'), pickup_count=('pickup_count', 'sum'))
        )
        part_stats['error_rate'] = part_stats.apply(
            lambda row: safe_div(row['error_count'], row['pickup_count']), axis=1
        )
        top_machine_per_part = (
            comp_filtered.sort_values('error_count', ascending=False)
            .drop_duplicates('part_number')[['part_number', 'machine_id']]
        )
        parts_view = (
            part_stats.merge(top_machine_per_part, on='part_number', how='left')
            .sort_values('error_rate', ascending=False)
            .head(10)
        )
        parts_view['error_rate'] = parts_view.apply(
            lambda row: f"{row['error_rate'] * 100:.1f}% ({int(row['error_count'])}/{int(row['pickup_count'])})", axis=1
        )
        parts_view = parts_view.rename(columns={'part_number': 'Part', 'machine_id': 'Top Machine'})
        parts_view = parts_view[['Part', 'Top Machine', 'error_rate']]
        col_right.markdown("#### Hot Parts (Top 10)")
        col_right.table(parts_view)

    st.text_area("Copy ", operator_summary, height=320)

    st.caption("※ 관측 기반 요약이며, 시간축 근사/누적 proxy 가능성이 있습니다.")

def _hms(sec: float) -> str:
    if sec is None or pd.isna(sec):
        return "-"
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h}:{m:02d}:{s:02d}"


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    st.set_page_config(layout='wide', page_title='SMT Process Analytics')
    st.title('Mount Manufacturing Analytics Demo')
    st.markdown('RCA Dashboard')

    #engine = get_engine()
    sample_mode = False
    try:
        raw_data = load_data()
    except Exception:
        raw_data = generate_sample_data()
        sample_mode = True

    if raw_data.get('_meta', {}).get('is_sample'):
        sample_mode = True
        st.markdown("<span style='color:#f5b642;'>Sample Data Mode</span>", unsafe_allow_html=True)

    base_data = {k: v for k, v in raw_data.items() if k != '_meta'}
    views = build_views(base_data)
    filters = collect_filters(views)
    filtered_views = {
        name: apply_filters(df, filters, name)
        for name, df in views.items()
    }

    render_poc_badge()
    kpis = calculate_kpis(filtered_views['vw_lot_machine_summary'], filtered_views['vw_stop_enriched'])
    render_kpi_cards(kpis)
    reliability = compute_reliability_indicators(filtered_views['vw_stop_enriched'])
    render_reliability_badge(reliability)

    #exec_summary, operator_summary = generate_rca_summary(filtered_views, filters)
    #if st.button('Generate RCA Summary'):
    #    st.code(exec_summary, language='text')
    #    st.code(operator_summary, language='text')
    #    render_action_templates(extract_top_stop_reasons(filtered_views['vw_stop_enriched']))

    tab1, tab2, tab3, tab4 = st.tabs(["RCA Summary", "Process View", "Machine View", "Quality View"])

    with tab1:
        render_rca_summary(filtered_views, filters)
    with tab2:
        render_process_view(filtered_views, filters)
    with tab3:
        render_machine_view(filtered_views, filters)
    with tab4:
        render_quality_view(filtered_views, filters)


if __name__ == '__main__':
    main()
