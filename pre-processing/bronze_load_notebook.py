# Databricks notebook source
# MAGIC %md
# MAGIC # 01_load_bronze
# MAGIC Loads the four raw files into bronze **exactly as given**: every column as text, no cleaning,
# MAGIC no rows dropped. Adds lineage columns, checks row and column counts, and records each load
# MAGIC in `audit.bronze_load_log`.
# MAGIC
# MAGIC **Before running:** upload the raw files to `/Volumes/datathon/bronze/raw_files/` and set the
# MAGIC file names in the next cell. If a bronze table already exists and passes the checks, it is
# MAGIC left alone unless `REPLACE_EXISTING = True`.

# COMMAND ----------

CATALOG = "datathon"            # or "workspace" if you couldn't create a catalog
RAW = f"/Volumes/{CATALOG}/bronze/raw_files"
REPLACE_EXISTING = False        # True = reload even if the existing table passes the checks

SOURCES = [
    # table name,      file in the volume,                         expected rows, expected columns, first column
    ("vf_facilities",  "New_Query_2026_11_21_17_17_54.csv",         10088,  51,  "unique_id"),
    ("blood_banks",    "India Blood Bank Directory (CSV).csv",      2823,   15,  "state"),
    ("pincode",        "india_post_pincode_directory.csv",          165627, 11,  "circlename"),
    ("nfhs5",          "nfhs5_district_health_indicators.csv",      706,    109, None),
]

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.bronze")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.audit")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.bronze.raw_files")
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.audit.bronze_load_log (
  table_name STRING, source_file STRING, file_md5 STRING, expected_rows BIGINT, loaded_rows BIGINT,
  loaded_columns INT, checks_passed BOOLEAN, check_notes STRING, loaded_at TIMESTAMP, loaded_by STRING)
""")

# COMMAND ----------

import hashlib, os

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def check(table, exp_rows, exp_cols, first_col):
    """Return (passed, notes). Lineage columns (starting with _) are not counted."""
    df = spark.table(table)
    cols = [c for c in df.columns if not c.startswith("_")]
    n = df.count()
    notes = []
    if n != exp_rows:
        notes.append(f"rows {n} != {exp_rows}")
    if len(cols) != exp_cols:
        notes.append(f"columns {len(cols)} != {exp_cols}")
    if first_col and cols and cols[0] != first_col:
        notes.append(f"first column '{cols[0]}' != '{first_col}' (encoding marker or renamed column?)")
    bad_types = [f.name for f in df.schema.fields if not f.name.startswith("_") and f.dataType.typeName() != "string"]
    if bad_types:
        notes.append(f"non-text columns: {bad_types[:5]}")
    return (not notes), "; ".join(notes) or "ok", n, len(cols)

def load(name, file, exp_rows, exp_cols, first_col):
    table, path = f"{CATALOG}.bronze.{name}", f"{RAW}/{file}"
    if not os.path.exists(path):
        print(f"SKIP {name}: file not found at {path}")
        return
    if spark.catalog.tableExists(table) and not REPLACE_EXISTING:
        ok, notes, n, c = check(table, exp_rows, exp_cols, first_col)
        if ok:
            print(f"KEEP {table}: already loaded and passes checks ({n} rows)")
            return
        print(f"RELOAD {table}: existing table fails checks -> {notes}")
    spark.sql(f"""
        CREATE OR REPLACE TABLE {table}
        TBLPROPERTIES ('delta.columnMapping.mode' = 'name')   -- allows NFHS-style column names
        AS SELECT *,
                  _metadata.file_path AS _source_file,
                  current_timestamp() AS _loaded_at
        FROM read_files('{path}',
             format => 'csv', header => true, multiLine => true,
             quote => '"', escape => '"', encoding => 'UTF-8',
             inferColumnTypes => false)                         -- every column stays text
    """)
    # a UTF-8 byte-order mark can end up in the first column name; strip it without touching data
    first = spark.table(table).columns[0]
    if first.startswith("﻿"):
        spark.sql(f"ALTER TABLE {table} RENAME COLUMN `{first}` TO `{first.lstrip(chr(0xFEFF))}`")
    ok, notes, n, c = check(table, exp_rows, exp_cols, first_col)
    spark.sql(f"COMMENT ON TABLE {table} IS 'Raw copy of {file}. All columns text, nothing cleaned. Loaded by 01_load_bronze.'")
    spark.createDataFrame(
        [(table, file, md5(path), exp_rows, n, c, ok, notes, spark.sql("SELECT current_user()").first()[0])],
        "table_name string, source_file string, file_md5 string, expected_rows long, loaded_rows long, "
        "loaded_columns int, checks_passed boolean, check_notes string, loaded_by string"
    ).selectExpr("table_name", "source_file", "file_md5", "expected_rows", "loaded_rows", "loaded_columns",
                 "checks_passed", "check_notes", "current_timestamp() AS loaded_at", "loaded_by") \
     .write.mode("append").saveAsTable(f"{CATALOG}.audit.bronze_load_log")
    print(("OK   " if ok else "FAIL ") + f"{table}: {n} rows, {c} columns -> {notes}")

for s in SOURCES:
    load(*s)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extra checks for the VF table
# MAGIC These describe the **source**, not a load error: the original file already contains 30 facilities
# MAGIC split across 88 fragment rows and 11 exact duplicates. Silver repairs and logs them.

# COMMAND ----------

display(spark.sql(f"""
SELECT
  COUNT(*)                                                                    AS bronze_records,        -- 10088
  SUM(CASE WHEN unique_id RLIKE '^[0-9a-f]{{8}}-[0-9a-f]{{4}}-' THEN 1 ELSE 0 END) AS facility_rows,    -- 10000
  SUM(CASE WHEN unique_id NOT RLIKE '^[0-9a-f]{{8}}-[0-9a-f]{{4}}-' THEN 1 ELSE 0 END) AS fragment_rows, -- 88
  COUNT(DISTINCT CASE WHEN unique_id RLIKE '^[0-9a-f]{{8}}-' THEN unique_id END) AS unique_facilities,  -- 9989
  SUM(CASE WHEN description LIKE '%\\n%' THEN 1 ELSE 0 END)                    AS multiline_descriptions -- > 0
FROM {CATALOG}.bronze.vf_facilities
"""))

# COMMAND ----------

display(spark.table(f"{CATALOG}.audit.bronze_load_log").orderBy("loaded_at", ascending=False))
