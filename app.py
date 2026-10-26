"""Dashboard Final Project: permintaan Yellow Taxi NYC 2025, prediksi satu jam ke depan, dan jaringan mobilitas.

Semua angka berasal dari satu run pipeline yang terverifikasi (lihat data/meta.json) dan sama dengan laporan.
Jalankan lokal:  streamlit run app.py
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas.tseries.holiday import USFederalHolidayCalendar

DATA = Path(__file__).parent / 'data'
NYC = {'lat': 40.705, 'lon': -73.94}
METHODS = {'sklearn_hgb': 'Model scikit-learn', 'spark_gbt': 'Model Spark MLlib',
           'last_hour': 'Baseline: jam sebelumnya', 'four_week_mean': 'Baseline: rata-rata 4 minggu',
           'last_week': 'Baseline: minggu lalu'}
COLORS = {'Aktual': '#0E1726', 'Model scikit-learn': '#E0A800', 'Model Spark MLlib': '#24649E',
          'Baseline: jam sebelumnya': '#5B6475', 'Baseline: rata-rata 4 minggu': '#A3ABB9',
          'Baseline: minggu lalu': '#C9CED8'}
AIRPORTS = {132: 'JFK Airport', 138: 'LaGuardia Airport'}

st.set_page_config(page_title='NYC Taxi 2025 · Dashboard', page_icon='🚕', layout='wide')


# ---------- formatting (Indonesian numbers) ----------

def num(value, places=0):
    text = f'{value:,.{places}f}'
    return text.replace(',', '_').replace('.', ',').replace('_', '.')


def millions(value):
    return f'{num(value / 1e6, 1)} jt'


def kpi(column, label, value, note=None):
    column.metric(label, value, border=True)
    if note:
        column.caption(note)


def style(fig, height=420):
    if fig.layout.title.text:
        fig.update_layout(title=dict(y=0.98, yanchor='top'))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=90 if fig.layout.title.text else 40, b=10),
                      separators=',.', legend=dict(orientation='h', yanchor='bottom', y=1.0, x=0, title=None),
                      font=dict(family='sans-serif', size=14))
    return fig


# ---------- data ----------

@st.cache_data
def load():
    meta = json.loads((DATA / 'meta.json').read_text())
    geo = json.loads((DATA / 'zones.geojson').read_text())
    zones = pd.read_csv(DATA / 'zones.csv')
    zones['wape_sklearn'] = np.where(zones.test_trips > 0,
                                     100 * zones.mae_sklearn_hgb * 1292 / zones.test_trips.replace(0, np.nan), np.nan)
    predictions = pd.read_parquet(DATA / 'predictions.parquet')
    daily_mae = pd.read_csv(DATA / 'daily_mae.csv', parse_dates=['date'])
    return {'meta': meta, 'geo': geo, 'zones': zones, 'predictions': predictions,
            'monthly': pd.read_csv(DATA / 'monthly.csv'),
            'daily_city': pd.read_csv(DATA / 'daily_city.csv', parse_dates=['date']),
            'profile': pd.read_parquet(DATA / 'week_profile.parquet'),
            'metrics': pd.read_csv(DATA / 'metrics.csv', index_col='method'),
            'bands': pd.read_csv(DATA / 'error_by_band.csv', dtype={'band': str}),
            'daily_mae': daily_mae, 'edges': pd.read_parquet(DATA / 'edges.parquet')}


def zone_map(zones, geo, column, title, height=520, **kwargs):
    fig = px.choropleth_map(zones, geojson=geo, locations='zone', color=column, hover_name='Zone',
                            hover_data={'zone': False, 'Borough': True, column: ':,.0f'},
                            map_style='carto-positron', center=NYC, zoom=9.4, opacity=0.8, **kwargs)
    fig.update_layout(title=title, coloraxis_colorbar=dict(title=None))
    return map_margin(style(fig, height))


def map_margin(fig):
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=40))  # room for the map attribution under the tiles
    return fig


def period_flows(edges, zones, period):
    """Departures, arrivals and net flow per zone for the whole year, weekdays or weekends."""
    if period != 'Tahunan':
        edges = edges.loc[edges.weekend == (period == 'Weekend')]
    out = edges.groupby('origin').trips.sum()
    arrivals = edges.groupby('destination').trips.sum()
    frame = zones[['zone', 'Zone', 'Borough']].copy()
    frame['departures'] = frame.zone.map(out).fillna(0)
    frame['arrivals'] = frame.zone.map(arrivals).fillna(0)
    frame['net_flow'] = frame.arrivals - frame.departures
    return edges, frame


def holidays():
    return USFederalHolidayCalendar().holidays('2025-11-01', '2025-12-31')


d = load()
meta, zones, metrics = d['meta'], d['zones'], d['metrics']
best, hour = metrics.loc['sklearn_hgb', 'mae'], metrics.loc['last_hour', 'mae']

st.sidebar.title('🚕 NYC Taxi 2025')
page = st.sidebar.radio('Halaman', ['Ringkasan', 'Prediksi Demand', 'Jaringan Mobilitas', 'Insight Bisnis'])
st.sidebar.markdown('---')
st.sidebar.caption(f"Data: NYC TLC Yellow Taxi Jan–Des 2025. Hasil dari run pipeline `{meta['run']}` "
                   '(Spark + MinIO + Neo4j), sama dengan laporan.')


# ---------- page 1: overview ----------

if page == 'Ringkasan':
    st.title('Permintaan Yellow Taxi New York City 2025')
    st.markdown('Berapa banyak pickup di setiap zona, kapan ramainya, dan seberapa akurat prediksi satu jam ke depan.')
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, 'Trip bersih 2025', millions(meta['clean_trips']), f"dari {millions(meta['raw_trips'])} trip mentah")
    kpi(c2, 'Zona taksi', '263', '5 borough dan bandara')
    kpi(c3, 'Error prediksi', num(best, 2), 'MAE, pickup per zona per jam')
    kpi(c4, 'Lebih akurat', f'{num(100 * (1 - best / hour), 1)}%',
        f'dibanding "sama dengan jam sebelumnya" (MAE {num(hour, 2)})')

    left, right = st.columns([3, 2])
    with left:
        cap = float(zones.pickups_2025.quantile(0.95))
        st.plotly_chart(zone_map(zones, d['geo'], 'pickups_2025', 'Total pickup per zona, 2025',
                                 color_continuous_scale='YlOrBr', range_color=(0, cap)))
        st.caption('Warna dibatasi pada persentil ke-95 supaya zona di luar Manhattan tetap terlihat.')
    with right:
        monthly = d['monthly'].assign(Bulan=lambda m: m.source_month.str[-2:])
        fig = px.bar(monthly, x='Bulan', y='clean_trips', title='Trip bersih per bulan',
                     labels={'clean_trips': 'Trip'}, color_discrete_sequence=['#0E1726'])
        st.plotly_chart(style(fig, 260))
        profile = d['profile'].groupby(['day_of_week', 'hour']).trips.sum().unstack()
        days = ['Sen', 'Sel', 'Rab', 'Kam', 'Jum', 'Sab', 'Min']
        fig = px.imshow(profile, labels=dict(x='Jam', y='Hari', color='Pickup/jam'), y=days,
                        color_continuous_scale='YlOrBr', aspect='auto', title='Rata-rata pickup per jam (seluruh kota)')
        st.plotly_chart(style(fig, 260))

    city = d['daily_city']
    fig = px.line(city, x='date', y='trips', title='Pickup harian seluruh kota',
                  labels={'date': '', 'trips': 'Pickup'}, color_discrete_sequence=['#24649E'])
    st.plotly_chart(style(fig, 300))


# ---------- page 2: forecasting ----------

elif page == 'Prediksi Demand':
    st.title('Prediksi pickup satu jam ke depan')
    st.markdown('Model hanya memakai data sampai akhir jam berjalan. Dievaluasi pada **November–Desember 2025** '
                '(data test yang tidak dipakai untuk memilih model).')
    order = ['sklearn_hgb', 'spark_gbt', 'last_hour', 'four_week_mean', 'last_week']
    table = metrics.loc[order].assign(Metode=lambda m: m.index.map(METHODS))
    fig = px.bar(table, x='mae', y='Metode', orientation='h', color='Metode', color_discrete_map=COLORS,
                 text=table.mae.map(lambda v: num(v, 3)), title='MAE test: rata-rata selisih pickup per zona-jam',
                 labels={'mae': 'MAE (lebih kecil lebih baik)', 'Metode': ''})
    fig.update_layout(showlegend=False, yaxis=dict(autorange='reversed'))
    st.plotly_chart(style(fig, 320))

    st.subheader('Jelajahi satu zona')
    test_totals = zones.loc[zones.test_trips > 0].sort_values('test_trips', ascending=False)
    names = test_totals.Zone.tolist()
    c1, c2, c3 = st.columns([2, 2, 3])
    zone_name = c1.selectbox('Zona (urut dari yang paling ramai)', names, index=names.index('Upper East Side South'))
    period = c2.date_input('Periode', (pd.Timestamp('2025-12-08'), pd.Timestamp('2025-12-14')),
                           min_value=pd.Timestamp('2025-11-01'), max_value=pd.Timestamp('2025-12-31'))
    start, end = (period[0], period[-1]) if isinstance(period, (tuple, list)) and period else (period, period)
    chosen = c3.multiselect('Metode', list(METHODS.values()),
                            default=['Model scikit-learn', 'Model Spark MLlib', 'Baseline: jam sebelumnya'])
    zone_id = int(zones.loc[zones.Zone == zone_name, 'zone'].iloc[0])
    p = d['predictions']
    window = p.loc[(p.zone == zone_id) & (p.pickup_hour >= pd.Timestamp(start))
                   & (p.pickup_hour < pd.Timestamp(end) + pd.Timedelta(days=1))]
    hours = pd.DataFrame({'pickup_hour': pd.date_range(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23),
                                                       freq='h')})
    window = hours.merge(window, on='pickup_hour', how='left')  # gaps (DST rule) stay visible as breaks
    long = window.rename(columns={'trips': 'Aktual', **METHODS})[['pickup_hour', 'Aktual', *chosen]].melt(
        'pickup_hour', var_name='Seri', value_name='Pickup')
    fig = px.line(long, x='pickup_hour', y='Pickup', color='Seri', color_discrete_map=COLORS,
                  title=f'{zone_name}: aktual vs prediksi per jam', labels={'pickup_hour': ''})
    fig.update_traces(selector=dict(name='Aktual'), line=dict(width=3))
    st.plotly_chart(style(fig, 420))
    scored = window.dropna(subset=['trips'])
    if len(scored):
        cols = st.columns(len(chosen) or 1)
        reverse = {v: k for k, v in METHODS.items()}
        for col, label in zip(cols, chosen):
            col.metric(label, f'MAE {num(float(np.abs(scored[reverse[label]] - scored.trips).mean()), 2)}')
    else:
        st.info('Tidak ada jam test pada periode ini (1–9 November sebagian dikecualikan karena aturan daylight saving).')

    left, right = st.columns(2)
    with left:
        bands = d['bands'].loc[d['bands'].method.isin(['sklearn_hgb', 'spark_gbt', 'last_hour'])].copy()
        bands['Metode'] = bands.method.map(METHODS)
        fig = px.bar(bands, x='band', y='mae', color='Metode', barmode='group', color_discrete_map=COLORS,
                     category_orders={'band': ['0', '1–10', '11–100', '>100']},
                     labels={'band': 'Pickup aktual per zona-jam', 'mae': 'MAE'},
                     title='Error menurut tingkat keramaian zona')
        st.plotly_chart(style(fig, 400))
    with right:
        metric = st.radio('Peta error per zona', ['Error absolut (MAE)', 'Error relatif (WAPE %)'], horizontal=True)
        column = 'wape_sklearn' if metric.startswith('Error relatif') else 'mae_sklearn_hgb'
        cap = float(zones[column].quantile(0.9))
        fig = zone_map(zones, d['geo'], column, 'Error model scikit-learn per zona',
                       height=400, color_continuous_scale='Reds', range_color=(0, cap))
        st.plotly_chart(fig)
        if column == 'wape_sklearn':
            st.caption('Zona sepi terlihat merah karena volumenya kecil: meleset 1 pickup sudah besar secara persen.')

    daily = d['daily_mae'].loc[lambda x: x.hours >= 20]
    long = daily.melt('date', ['sklearn_hgb', 'spark_gbt', 'last_hour', 'last_week'], var_name='m', value_name='MAE')
    long['Metode'] = long.m.map(METHODS)
    fig = px.line(long, x='date', y='MAE', color='Metode', color_discrete_map=COLORS,
                  title='Error harian (garis putus: hari libur federal)', labels={'date': ''})
    for day in holidays():
        fig.add_vline(x=day, line_dash='dot', line_color='#C0392B')
    st.plotly_chart(style(fig, 360))
    top = daily.nlargest(6, 'sklearn_hgb')[['date', 'sklearn_hgb', 'last_hour', 'last_week']]
    top.columns = ['Tanggal', 'MAE model', 'MAE jam sebelumnya', 'MAE minggu lalu']
    st.markdown('**Hari dengan error model tertinggi**')
    st.dataframe(top.style.format({'Tanggal': lambda t: t.strftime('%a %d %b %Y'), 'MAE model': '{:.2f}',
                                   'MAE jam sebelumnya': '{:.2f}', 'MAE minggu lalu': '{:.2f}'}), hide_index=True)


# ---------- page 3: graph ----------

elif page == 'Jaringan Mobilitas':
    st.title('Jaringan perjalanan antarzona')
    st.markdown('Setiap zona adalah node; setiap pasangan asal → tujuan adalah edge berarah dengan bobot jumlah trip. '
                f"Graph dimuat ke Neo4j dan {'semua 6 pemeriksaan Cypher cocok' if meta['neo4j_checks_passed'] else 'ada pemeriksaan yang gagal'} "
                'dengan hasil NetworkX.')
    period = st.radio('Periode', ['Tahunan', 'Weekday', 'Weekend'], horizontal=True)
    edges, flows = period_flows(d['edges'], zones, period)
    g = meta['graph']
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, 'Trip', millions(edges.trips.sum()), 'dengan asal dan tujuan valid')
    kpi(c2, 'Pasangan zona', num(edges.groupby(['origin', 'destination']).ngroups), 'asal → tujuan berarah')
    kpi(c3, 'Komunitas', str(g['communities']), f"Louvain tahunan, modularity {num(g['modularity'], 3)}")
    manhattan = flows.loc[flows.Borough == 'Manhattan', 'arrivals'].sum() / flows.arrivals.sum()
    kpi(c4, 'Ke Manhattan', f'{num(100 * manhattan, 1)}%', 'dari semua kedatangan')

    view = st.selectbox('Tampilkan di peta', ['Arus bersih (kedatangan − keberangkatan)', 'Keberangkatan',
                                              'Kedatangan', 'PageRank (kepentingan zona tujuan)', 'Komunitas Louvain'])
    mapped = flows.merge(zones[['zone', 'pagerank', 'pagerank_weekday', 'pagerank_weekend', 'community']], on='zone')
    if view.startswith('Arus'):
        limit = float(np.nanpercentile(np.abs(mapped.net_flow), 98))
        fig = zone_map(mapped, d['geo'], 'net_flow', f'Arus bersih per zona ({period})',
                       color_continuous_scale='RdBu', range_color=(-limit, limit), color_continuous_midpoint=0)
        st.caption('Merah = lebih banyak berangkat daripada datang (pengirim). Biru = lebih banyak datang (penerima).')
    elif view == 'Keberangkatan':
        fig = zone_map(mapped, d['geo'], 'departures', f'Keberangkatan per zona ({period})',
                       color_continuous_scale='YlOrBr', range_color=(0, float(mapped.departures.quantile(0.95))))
    elif view == 'Kedatangan':
        fig = zone_map(mapped, d['geo'], 'arrivals', f'Kedatangan per zona ({period})',
                       color_continuous_scale='Blues', range_color=(0, float(mapped.arrivals.quantile(0.95))))
    elif view.startswith('PageRank'):
        column = {'Tahunan': 'pagerank', 'Weekday': 'pagerank_weekday', 'Weekend': 'pagerank_weekend'}[period]
        fig = px.choropleth_map(mapped, geojson=d['geo'], locations='zone', color=column, hover_name='Zone',
                                map_style='carto-positron', center=NYC, zoom=9.4, opacity=0.8,
                                color_continuous_scale='Viridis', title=f'PageRank ({period})')
        fig = map_margin(style(fig, 520))
    else:
        mapped['Komunitas'] = mapped.community.astype('Int64').astype(str)
        fig = px.choropleth_map(mapped, geojson=d['geo'], locations='zone', color='Komunitas', hover_name='Zone',
                                map_style='carto-positron', center=NYC, zoom=9.4, opacity=0.8,
                                title='Komunitas Louvain (graph tahunan)',
                                color_discrete_sequence=['#24649E', '#E0A800', '#C0392B', '#8A94A6', '#5B6475'])
        fig = map_margin(style(fig, 520))
    st.plotly_chart(fig)

    left, right = st.columns(2)
    names = zones.set_index('zone').Zone
    with left:
        corridors = (edges.loc[edges.origin != edges.destination].groupby(['origin', 'destination']).trips.sum()
                     .nlargest(10).reset_index())
        corridors['Asal'] = corridors.origin.map(names)
        corridors['Tujuan'] = corridors.destination.map(names)
        st.markdown(f'**10 koridor terbesar ({period})**')
        st.dataframe(corridors[['Asal', 'Tujuan', 'trips']].rename(columns={'trips': 'Trip'})
                     .style.format({'Trip': lambda v: num(v)}), hide_index=True)
    with right:
        boroughs = zones.set_index('zone').Borough
        airport = edges.loc[edges.origin.isin(AIRPORTS)].assign(
            Bandara=lambda e: e.origin.map(AIRPORTS), Tujuan=lambda e: e.destination.map(boroughs))
        share = airport.groupby(['Bandara', 'Tujuan']).trips.sum().reset_index()
        share['Persen'] = 100 * share.trips / share.groupby('Bandara').trips.transform('sum')
        fig = px.bar(share, x='Persen', y='Bandara', color='Tujuan', orientation='h',
                     title=f'Ke mana trip dari bandara berakhir ({period})', labels={'Persen': '% trip'},
                     category_orders={'Tujuan': ['Manhattan', 'Queens', 'Brooklyn', 'Bronx', 'Staten Island', 'EWR']},
                     color_discrete_map={'Manhattan': '#E0A800', 'Queens': '#24649E', 'Brooklyn': '#5B6475',
                                         'Bronx': '#A3ABB9', 'Staten Island': '#C9CED8', 'EWR': '#0E1726'})
        st.plotly_chart(style(fig, 330))

    st.markdown('**Zona penting yang paling naik dan turun peringkat PageRank-nya saat weekend** '
                '(zona di 40 besar weekday atau weekend)')
    ranks = zones.assign(weekday=zones.pagerank_weekday.rank(ascending=False, method='min'),
                         weekend=zones.pagerank_weekend.rank(ascending=False, method='min'))
    ranks = ranks.loc[(ranks.weekday <= 40) | (ranks.weekend <= 40)]
    up, down = st.columns(2)
    show = lambda frame: frame[['Zone', 'Borough', 'weekday', 'weekend']].rename(
        columns={'Zone': 'Zona', 'weekday': 'Peringkat weekday', 'weekend': 'Peringkat weekend'}).astype(
        {'Peringkat weekday': int, 'Peringkat weekend': int})
    up.markdown('Naik saat weekend')
    up.dataframe(show(ranks.assign(shift=ranks.weekday - ranks.weekend).nlargest(8, 'shift')), hide_index=True)
    down.markdown('Turun saat weekend')
    down.dataframe(show(ranks.assign(shift=ranks.weekday - ranks.weekend).nsmallest(8, 'shift')), hide_index=True)


# ---------- page 4: business insights ----------

else:
    st.title('Insight bisnis')
    st.markdown('Empat temuan untuk operator armada dan perencana transportasi, masing-masing dengan angka '
                'pendukung dan rekomendasi. Detailnya bisa dijelajahi di halaman lain.')
    bands = d['bands'].set_index(['band', 'method'])
    busy_wape = bands.loc[('>100', 'sklearn_hgb'), 'wape_pct']

    st.header('1 · Prediksi cukup andal untuk mengarahkan armada lebih awal')
    left, right = st.columns([2, 3])
    kpi(left, 'Lebih akurat', f'{num(100 * (1 - best / hour), 1)}%', 'dibanding "sama dengan jam sebelumnya"')
    kpi(left, 'Error di zona tersibuk', f'{num(busy_wape, 1)}%', 'dari volume, zona-jam dengan >100 pickup')
    order = ['sklearn_hgb', 'last_hour', 'last_week']
    fig = px.bar(metrics.loc[order].assign(Metode=lambda m: m.index.map(METHODS)), x='Metode', y='mae', color='Metode',
                 color_discrete_map=COLORS, text_auto='.2f', labels={'mae': 'MAE', 'Metode': ''})
    fig.update_layout(showlegend=False)
    right.plotly_chart(style(fig, 300))
    st.markdown(f'**Arti bisnis.** Rata-rata prediksi meleset {num(best, 1)} pickup per zona per jam. Di zona yang '
                f'paling penting secara bisnis, yaitu zona ramai, error-nya hanya sekitar {num(busy_wape, 0)}% dari '
                'volume. **Rekomendasi:** pakai prediksi untuk memberi tahu pengemudi atau dispatcher zona mana yang '
                'akan ramai satu jam lagi, terutama di Manhattan, supaya armada sudah ada sebelum penumpang menunggu.')

    st.header('2 · Hari khusus perlu penanganan tambahan')
    daily = d['daily_mae'].loc[lambda x: x.hours >= 20]
    top = daily.nlargest(5, 'sklearn_hgb')
    fig = px.line(daily, x='date', y=['sklearn_hgb', 'last_week'], labels={'date': '', 'value': 'MAE'},
                  color_discrete_sequence=['#E0A800', '#C9CED8'])
    fig.for_each_trace(lambda t: t.update(name=METHODS[t.name]))
    fig.add_trace(go.Scatter(x=top.date, y=top.sklearn_hgb, mode='markers+text', name='Hari tersulit',
                             text=top.date.dt.strftime('%d %b'), textposition='top center',
                             marker=dict(size=10, color='#C0392B')))
    st.plotly_chart(style(fig, 330))
    thanks = daily.set_index('date').loc[pd.Timestamp('2025-11-27')]
    st.markdown('**Arti bisnis.** Error tertinggi muncul sesudah Halloween, malam sebelum Thanksgiving, malam '
                'Natal, malam tahun baru, dan seminggu setelah Thanksgiving, yaitu hari ketika pola perjalanan berubah. Fitur hari libur sudah '
                f"membantu (Thanksgiving: MAE model {num(thanks.sklearn_hgb, 2)} vs {num(thanks.last_week, 2)} untuk "
                'minggu lalu), tetapi malam acara besar tetap sulit. **Rekomendasi:** siapkan kalender acara dan buffer '
                'armada untuk tanggal-tanggal ini, dan jangan bergantung penuh pada prediksi otomatis.')

    st.header('3 · Bandara adalah pengirim taksi, bukan penerima')
    airports = zones.loc[zones.zone.isin(AIRPORTS)].set_index('Zone')
    left, right = st.columns([2, 3])
    for name, row in airports.iterrows():
        kpi(left, f'Arus bersih {name}', f'{num(row.net_flow / 1e6, 2)} jt',
            f"{num(row.out_strength)} berangkat · {num(row.in_strength)} datang per tahun")
    shares = meta['airport_manhattan_share_pct']
    fig = px.bar(airports.reset_index(), x='Zone', y=['out_strength', 'in_strength'], barmode='group',
                 labels={'value': 'Trip per tahun', 'Zone': ''}, color_discrete_sequence=['#C0392B', '#24649E'])
    fig.for_each_trace(lambda t: t.update(name={'out_strength': 'Berangkat', 'in_strength': 'Datang'}[t.name]))
    right.plotly_chart(style(fig, 300))
    st.markdown(f"**Arti bisnis.** Dari JFK jauh lebih banyak taksi berangkat membawa penumpang daripada yang datang. "
                f"{num(shares['JFK Airport'], 1)}% trip dari JFK dan {num(shares['LaGuardia Airport'], 1)}% dari "
                'LaGuardia berakhir di Manhattan, sehingga taksi menumpuk di Manhattan sementara bandara terus '
                'membutuhkan pasokan. **Rekomendasi:** rencanakan arus balik ke bandara, misalnya antrean atau '
                'insentif trip menuju bandara. Catatan: data hanya mencatat trip berpenumpang, jadi perjalanan '
                'kosong tidak terukur langsung.')

    st.header('4 · Manhattan pusat jaringan, tetapi weekend berbeda')
    ranks = zones.assign(weekday=zones.pagerank_weekday.rank(ascending=False, method='min'),
                         weekend=zones.pagerank_weekend.rank(ascending=False, method='min'))
    left, right = st.columns(2)
    left.markdown('**10 besar PageRank weekday**')
    left.dataframe(ranks.nsmallest(10, 'weekday')[['Zone', 'weekday', 'weekend']].astype(
        {'weekday': int, 'weekend': int}).rename(columns={'Zone': 'Zona', 'weekday': 'Weekday', 'weekend': 'Weekend'}),
        hide_index=True)
    right.markdown('**10 besar PageRank weekend**')
    right.dataframe(ranks.nsmallest(10, 'weekend')[['Zone', 'weekend', 'weekday']].astype(
        {'weekday': int, 'weekend': int}).rename(columns={'Zone': 'Zona', 'weekday': 'Weekday', 'weekend': 'Weekend'}),
        hide_index=True)
    _, flows = period_flows(d['edges'], zones, 'Tahunan')
    manhattan = flows.loc[flows.Borough == 'Manhattan', 'arrivals'].sum() / flows.arrivals.sum()
    st.markdown(f'**Arti bisnis.** {num(100 * manhattan, 1)}% kedatangan berakhir di Manhattan, dengan Upper East '
                'Side dan Midtown sebagai pusat saat weekday. Saat weekend, zona hiburan seperti East Village naik ke '
                'peringkat atas, sedangkan pusat perkantoran Midtown turun. **Rekomendasi:** pakai pola penempatan '
                'armada yang berbeda untuk weekday (perkantoran dan perumahan) dan weekend (hiburan dan malam hari).')
