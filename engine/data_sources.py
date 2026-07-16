"""
data_sources.py
================
Helpers for sourcing real, publicly available EXPOSURE and VULNERABILITY
data to replace the synthetic examples. These functions require internet
access to the respective providers, which will typically not be available
inside a sandboxed execution environment - run this module in your own
Python environment.

EXPOSURE: building footprints
------------------------------
1) Overture Maps (buildings theme)
   Global, open (CDLA Permissive-2.0), released monthly as cloud-hosted
   GeoParquet. Queryable directly with DuckDB's spatial + httpfs
   extensions without downloading the full dataset - you filter by
   bounding box and only transfer what you need.
   Docs: https://docs.overturemaps.org/getting-data/duckdb/

2) Microsoft Global ML Building Footprints
   Global building footprints (polygon + confidence score) derived from
   satellite imagery, released as country-level GeoJSON/quantized-mesh
   downloads under ODbL.
   Repo: https://github.com/microsoft/GlobalMLBuildingFootprints

VULNERABILITY: flood depth-damage curves
------------------------------------------
JRC Global Flood Depth-Damage Functions (Huizinga, de Moel & Szewczyk,
2017). Open dataset of fractional damage curves by continent and
occupancy class (residential / commercial / industrial), plus
max-damage-per-m2 by country, published by the EU Joint Research Centre.
Landing page: https://publications.jrc.ec.europa.eu/repository/handle/JRC105688
DOI: 10.2760/16510
A pre-processed CSV extract is also maintained by the open-source
`physrisk` project: see physrisk docs for the parsed table.

Both data sources above are free to use; check each licence (CDLA
Permissive-2.0 for Overture, ODbL for Microsoft, EU JRC open data terms
for the vulnerability functions) before redistributing.
"""

from __future__ import annotations

import pandas as pd


OVERTURE_BUILDINGS_DUCKDB_TEMPLATE = """
-- Requires: pip install duckdb
INSTALL spatial; LOAD spatial;
INSTALL httpfs; LOAD httpfs;
SET s3_region='us-west-2';

COPY (
    SELECT
        id,
        subtype,
        class,
        height,
        num_floors,
        ST_X(ST_Centroid(geometry)) AS lon,
        ST_Y(ST_Centroid(geometry)) AS lat,
        ST_Area(ST_Transform(geometry, 'EPSG:4326', 'EPSG:3857')) AS area_m2
    FROM read_parquet(
        's3://overturemaps-us-west-2/release/{release}/theme=buildings/type=building/*',
        filename=true, hive_partitioning=1
    )
    WHERE bbox.xmin BETWEEN {lon_min} AND {lon_max}
      AND bbox.ymin BETWEEN {lat_min} AND {lat_max}
) TO '{output_path}' WITH (FORMAT PARQUET);
"""


def fetch_overture_buildings(lon_min: float, lat_min: float, lon_max: float, lat_max: float,
                              output_path: str = "overture_buildings.parquet",
                              release: str = "2026-06-17.0") -> str:
    """
    Build (and optionally run) the DuckDB query that pulls Overture Maps
    building footprints for a bounding box, with centroid lon/lat and
    footprint area already computed - ready to feed into
    `ExposurePortfolio.from_footprints`.

    Requires `duckdb` to be installed and outbound internet access to the
    Overture S3 bucket. Returns the SQL used (also written to a .sql file
    next to `output_path`) so it can be run separately if this
    environment has no internet access.
    """
    sql = OVERTURE_BUILDINGS_DUCKDB_TEMPLATE.format(
        release=release, lon_min=lon_min, lon_max=lon_max,
        lat_min=lat_min, lat_max=lat_max, output_path=output_path,
    )
    sql_path = output_path.rsplit(".", 1)[0] + ".sql"
    with open(sql_path, "w") as f:
        f.write(sql)

    try:
        import duckdb
        con = duckdb.connect()
        con.execute(sql)
        return output_path
    except Exception as exc:  # pragma: no cover - network/env dependent
        print(
            f"[data_sources] Could not run query automatically ({exc}).\n"
            f"Run the saved query manually: duckdb -c \".read {sql_path}\"\n"
            f"or from Python with `duckdb`, once you have internet access to "
            f"the Overture S3 bucket."
        )
        return sql_path


def microsoft_building_footprints_info(country_or_region: str) -> str:
    """
    Microsoft's Global ML Building Footprints are distributed as one
    dataset (GeoJSON/quantized-mesh, `.csv` link index) per
    country/region rather than a single queryable endpoint. Returns
    instructions for the requested region; download and convert to
    (id, lon, lat, area_m2, class) with geopandas before passing to
    `ExposurePortfolio.from_footprints`.
    """
    return (
        f"Microsoft Global ML Building Footprints for '{country_or_region}':\n"
        f"1. Browse the dataset links index: "
        f"https://github.com/microsoft/GlobalMLBuildingFootprints\n"
        f"2. Download the relevant country/region GeoJSON(.gz) file.\n"
        f"3. Load with geopandas, project to an equal-area CRS to compute "
        f"footprint area_m2, then take centroids for lon/lat:\n"
        f"     gdf = gpd.read_file(path)\n"
        f"     gdf['area_m2'] = gdf.geometry.to_crs('EPSG:6933').area\n"
        f"     centroids = gdf.geometry.centroid\n"
        f"     gdf['lon'], gdf['lat'] = centroids.x, centroids.y\n"
        f"4. Pass the resulting DataFrame to ExposurePortfolio.from_footprints()."
    )


JRC_VULNERABILITY_INFO = (
    "JRC Global Flood Depth-Damage Functions (Huizinga et al., 2017)\n"
    "Landing page: https://publications.jrc.ec.europa.eu/repository/handle/JRC105688 \n"
    "DOI: 10.2760/16510\n\n"
    "The published Excel workbook contains fractional-damage curves by "
    "continent and occupancy class (residential/commercial/industrial) "
    "plus max-damage-per-m2 by country. To use it in this engine:\n"
    "  1. Download the workbook from the JRC repository above.\n"
    "  2. Extract the (depth_m, damage_ratio) points for the "
    "continent/occupancy classes you need.\n"
    "  3. Multiply the fractional damage ratio by your local "
    "max-damage-per-m2 x footprint area to get monetary vulnerability, "
    "or simply use the fractional curve directly against asset `value` "
    "(this engine's default) as VulnerabilitySet.from_csv() expects: \n"
    "     occupancy_type, depth_m, damage_ratio, cv\n"
    "A pre-parsed CSV extract is also available via the open-source "
    "`physrisk` project's onboarding notebook (search 'physrisk JRC "
    "inundation onboarding')."
)


def print_data_source_guide() -> None:
    print(__doc__)
    print("\n" + JRC_VULNERABILITY_INFO)
