"""Build the small data files the Streamlit dashboard reads, from one completed pipeline run.

Reads only the run's own outputs (Gold tables, predictions, graph tables) plus the zone shapefile, and
writes `SourceCode/dashboard/data/`. The run folder is never modified. Totals are reconciled before
anything is written, so the dashboard shows exactly the numbers in the report.

Run from the project root with the dashboard environment:
    .venv-dashboard/Scripts/python.exe SourceCode/dashboard/prepare_dashboard_data.py
"""
from pathlib import Path
import io
import json
import zipfile

import numpy as np
import pandas as pd
import shapefile
from pyproj import CRS, Transformer

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / 'output/assignment_II/independent_pipeline'
ZONES_ZIP = ROOT / 'data/bronze/zones/taxi_zones.zip'
LOOKUP = ROOT / 'data/bronze/zones/taxi_zone_lookup.csv'
OUT = Path(__file__).resolve().parent / 'data'
METHODS = ['last_hour', 'last_week', 'four_week_mean', 'sklearn_hgb', 'spark_gbt']


def scores(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    error = np.clip(np.asarray(predicted, dtype=float), 0, None) - actual
    return {'mae': float(np.abs(error).mean()), 'rmse': float(np.sqrt((error ** 2).mean())),
            'wape_pct': float(100 * np.abs(error).sum() / actual.sum()) if actual.sum() > 0 else None,
            'bias': float(error.mean()), 'rows': int(len(actual))}


def zone_geojson(lookup):
    """Zone polygons in longitude/latitude, lightly thinned so the file stays small."""
    with zipfile.ZipFile(ZONES_ZIP) as archive:
        part = lambda suffix: io.BytesIO(archive.read(next(n for n in archive.namelist() if n.endswith(suffix))))
        wkt = archive.read(next(n for n in archive.namelist() if n.endswith('.prj'))).decode()
        reader = shapefile.Reader(shp=part('.shp'), shx=part('.shx'), dbf=part('.dbf'))
        to_lonlat = Transformer.from_crs(CRS.from_wkt(wkt), 'EPSG:4326', always_xy=True)
        polygons = {}
        for record in reader.iterShapeRecords():
            zone = int(record.record.as_dict()['LocationID'])
            points = np.asarray(record.shape.points)
            bounds = [*record.shape.parts, len(points)]
            for a, b in zip(bounds, bounds[1:]):
                ring = points[a:b]
                step = max(1, len(ring) // 150)
                ring = np.vstack([ring[::step], ring[:1]])  # keep the ring closed after thinning
                lon, lat = to_lonlat.transform(ring[:, 0], ring[:, 1])
                polygons.setdefault(zone, []).append([[round(x, 5), round(y, 5)] for x, y in zip(lon, lat)])
    names = lookup.set_index('zone').Zone
    features = [{'type': 'Feature', 'id': zone, 'properties': {'zone': zone, 'name': names.get(zone, '')},
                 'geometry': {'type': 'MultiPolygon', 'coordinates': [[ring] for ring in rings]}}
                for zone, rings in sorted(polygons.items())]
    return {'type': 'FeatureCollection', 'features': features}


def main():
    run = ROOT / json.loads((PIPELINE / 'latest.json').read_text())['run']
    if not (run / '_SUCCESS').exists():
        raise ValueError(f'{run.name} is not a completed run')
    OUT.mkdir(exist_ok=True)
    lookup = pd.read_csv(LOOKUP).rename(columns={'LocationID': 'zone'})
    lookup = lookup.loc[lookup.zone.between(1, 263)]

    # Volume and quality per month.
    quality = pd.read_csv(run / 'data_quality_by_month.csv')
    quality.to_csv(OUT / 'monthly.csv', index=False)

    # Demand: city-wide daily totals, zone totals and the average week profile per zone.
    demand = pd.read_parquet(run / 'gold_zone_hour_demand.parquet', columns=['zone', 'pickup_hour', 'trips'])
    if int(demand.trips.sum()) != int(quality.clean_trips.sum()):
        raise ValueError('Gold demand does not reconcile with the clean trip count')
    demand['date'] = demand.pickup_hour.dt.normalize()
    demand.groupby('date', as_index=False).trips.sum().to_csv(OUT / 'daily_city.csv', index=False)
    demand['day_of_week'] = demand.pickup_hour.dt.dayofweek
    demand['hour'] = demand.pickup_hour.dt.hour
    profile = demand.groupby(['zone', 'day_of_week', 'hour'], as_index=False).trips.mean()
    profile['trips'] = profile.trips.round(2).astype('float32')
    profile.to_parquet(OUT / 'week_profile.parquet', index=False)

    # Test predictions plus the "same as the previous hour" baseline from the run's own lag_1 feature.
    predictions = pd.read_parquet(run / 'test_predictions.parquet',
                                  columns=['zone', 'pickup_hour', 'trips', 'last_week', 'four_week_mean',
                                           'sklearn_hgb', 'spark_gbt'])
    design = pd.read_parquet(run / 'test_design.parquet', columns=['zone', 'pickup_hour', 'lag_1'])
    predictions = predictions.merge(design.rename(columns={'lag_1': 'last_hour'}), on=['zone', 'pickup_hour'],
                                    how='left', validate='one_to_one')
    if predictions.last_hour.isna().any():
        raise ValueError('lag_1 missing for some test rows')
    recorded = pd.read_csv(run / 'test_metrics.csv', index_col=0)
    for method in ['sklearn_hgb', 'spark_gbt', 'last_week']:
        if abs(scores(predictions.trips, predictions[method])['mae'] - recorded.loc[method, 'mae']) > 1e-9:
            raise ValueError(f'{method} MAE does not reproduce the recorded test metric')
    for column in METHODS:
        predictions[column] = np.clip(predictions[column], 0, None).astype('float32')
    predictions.to_parquet(OUT / 'predictions.parquet', index=False)

    metrics = pd.DataFrame({m: scores(predictions.trips, predictions[m]) for m in METHODS}).T
    metrics = metrics.join(recorded[['mae_low', 'mae_high']])
    metrics.index.name = 'method'
    metrics.to_csv(OUT / 'metrics.csv')

    bands = pd.cut(predictions.trips, [-1, 0, 10, 100, np.inf], labels=['0', '1–10', '11–100', '>100'])
    segments = pd.DataFrame([{'band': str(b), 'method': m, **scores(g.trips, g[m])}
                             for b, g in predictions.groupby(bands, observed=True) for m in METHODS])
    segments.to_csv(OUT / 'error_by_band.csv', index=False)

    day = predictions.pickup_hour.dt.normalize()
    daily = predictions.assign(date=day).groupby('date').apply(
        lambda g: pd.Series({'hours': g.pickup_hour.nunique(),
                             **{m: float(np.abs(g[m] - g.trips).mean()) for m in METHODS}}), include_groups=False)
    daily.reset_index().to_csv(OUT / 'daily_mae.csv', index=False)

    zone_error = predictions.groupby('zone').apply(
        lambda g: pd.Series({'test_trips': int(g.trips.sum()),
                             **{f'mae_{m}': float(np.abs(g[m] - g.trips).mean()) for m in METHODS}}),
        include_groups=False).reset_index()

    # Graph: nodes with names, weekday/weekend edges, communities.
    nodes = pd.read_parquet(run / 'gold_graph_nodes.parquet')
    edges = pd.read_parquet(run / 'gold_od_edges.parquet')
    summary = json.loads((run / 'graph_summary.json').read_text())
    if int(edges.trips.sum()) != summary['total_trips']:
        raise ValueError('OD edges do not reconcile with the graph total')
    edges.to_parquet(OUT / 'edges.parquet', index=False)
    totals = demand.groupby('zone').trips.sum().rename('pickups_2025')
    zones = (lookup[['zone', 'service_zone']].merge(nodes, on='zone', how='left', validate='one_to_one')
             .merge(totals, on='zone', how='left').merge(zone_error, on='zone', how='left'))
    zones.to_csv(OUT / 'zones.csv', index=False)

    (OUT / 'zones.geojson').write_text(json.dumps(zone_geojson(lookup)), encoding='utf-8')

    neo4j = json.loads((run / 'neo4j_verification.json').read_text())
    results = json.loads((run / 'results.json').read_text())
    meta = {'run': run.name, 'graph': summary, 'eda': results['eda'], 'timings': results['timings'],
            'spark_seed_scores': results['spark_seed_scores'], 'selection': results['selection'],
            'airport_manhattan_share_pct': neo4j['airport_manhattan_share_pct'],
            'neo4j_checks_passed': all(str(v) == 'True' for v in neo4j['checks'].values()),
            'raw_trips': int(quality.raw_trips.sum()), 'clean_trips': int(quality.clean_trips.sum()),
            'valid_od_trips': int(quality.valid_od_trips.sum())}
    (OUT / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    pd.read_csv(run / 'scaling.csv').to_csv(OUT / 'scaling.csv', index=False)

    size = sum(f.stat().st_size for f in OUT.iterdir()) / 1e6
    print(f'Dashboard data from run {run.name}: {len(list(OUT.iterdir()))} files, {size:.1f} MB')
    print(metrics[['mae', 'wape_pct']].round(3).to_string())


if __name__ == '__main__':
    main()
