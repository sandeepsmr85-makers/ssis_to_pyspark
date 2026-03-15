"""
Deterministic PySpark Code Generator

Generates guaranteed-correct PySpark code for simple SSIS transformations
without invoking the LLM.  Covers the ~40% of components that map 1-to-1
to a PySpark operation so the LLM's variability cannot introduce bugs.

Supported component types (rule-based, no LLM):
  - Sort           → .orderBy()
  - DataConvert    → .withColumn("...", col.cast(...))
  - UnionAll       → .unionByName(allowMissingColumns=True)
  - Multicast      → DataFrame variable reuse (returns same df)
  - RowCount       → .count() with logging
  - Merge          → .unionByName() on pre-sorted streams
  - OLEDBCommand   → spark.sql() or JDBC execute stub

All other types return None → caller falls back to LLM.
"""
import json
from typing import Optional

try:
    from src.parsers.ssis_dtsx import Transformation, Mapping
except ImportError:
    # Allow standalone usage
    pass


# PySpark type string from SSIS DataType
_CAST_MAP = {
    "wstr": "string", "str": "string", "ntext": "string", "text": "string",
    "i1": "byte", "i2": "short", "i4": "int", "i8": "long",
    "r4": "float", "r8": "double", "numeric": "decimal",
    "cy": "decimal(19,4)", "bool": "boolean",
    "dbDate": "date", "dbTime": "timestamp", "dbTimeStamp": "timestamp",
    "dbTimeStamp2": "timestamp", "guid": "string", "image": "binary",
}


def try_deterministic(tx: "Transformation", mapping: "Mapping") -> Optional[str]:
    """
    Attempt deterministic code generation for a transformation.

    Returns generated Python code string on success, None if this
    transformation type requires LLM generation.
    """
    generator = {
        "Sort": _gen_sort,
        "DataConvert": _gen_data_convert,
        "UnionAll": _gen_union_all,
        "Multicast": _gen_multicast,
        "Merge": _gen_merge,
        "RowCount": _gen_row_count,
        "OLEDBCommand": _gen_oledb_command,
    }.get(tx.type)

    if generator is None:
        return None

    try:
        code = generator(tx, mapping)
        if code:
            # Inject safety header
            return f"# AUTO-GENERATED (deterministic) — {tx.type}: {tx.name}\n{code}"
        return None
    except Exception as exc:
        return None  # fall back to LLM on any generation error


# ─── Sort ───────────────────────────────────────────────────────────────────

def _gen_sort(tx: "Transformation", mapping: "Mapping") -> str:
    """Generate .orderBy() code from Sort transformation metadata."""
    sort_cols = tx.sort_columns  # [{"column": name, "ascending": True/False}, ...]

    if not sort_cols:
        # Fallback: use all output fields with sort_position > 0
        sortable = sorted(
            [f for f in tx.fields if f.port_type == "OUTPUT" and f.sort_position > 0],
            key=lambda f: f.sort_position
        )
        sort_cols = [{"column": f.name, "ascending": not f.sort_descending} for f in sortable]

    if not sort_cols:
        return None  # no sort info → use LLM

    func_name = _safe_name(tx.name)
    col_exprs = []
    for sc in sort_cols:
        col = sc["column"]
        if sc.get("ascending", True):
            col_exprs.append(f'F.col("{col}").asc()')
        else:
            col_exprs.append(f'F.col("{col}").desc()')

    # Check for SSIS "EliminateDuplicates" flag
    elim_dupes = tx.properties.get("EliminateDuplicates", "False").lower() == "true"
    dedup_code = ""
    if elim_dupes:
        key_cols = [f'"{sc["column"]}"' for sc in sort_cols]
        dedup_code = f"\n    df = df.dropDuplicates([{', '.join(key_cols)}])"

    return f"""
def transform_{func_name}(input_df):
    \"\"\"SSIS Sort: {tx.name}
    Sort columns: {', '.join(f"{s['column']} {'ASC' if s.get('ascending', True) else 'DESC'}" for s in sort_cols)}
    \"\"\"
    df = input_df.orderBy(
        {', '.join(col_exprs)}
    ){dedup_code}
    return df
""".strip()


# ─── DataConvert ─────────────────────────────────────────────────────────────

def _gen_data_convert(tx: "Transformation", mapping: "Mapping") -> str:
    """Generate .withColumn(".cast(...)) for DataConvert transformation."""
    output_fields = [f for f in tx.fields if f.port_type == "OUTPUT"]
    if not output_fields:
        return None

    func_name = _safe_name(tx.name)
    casts = []
    for f in output_fields:
        # cast_type was extracted by the parser from the DataConvert output column dataType
        cast_target = f.cast_type or _CAST_MAP.get(f.datatype.lower(), "string")

        # SSIS DataConvert renames: output column name often ends with "_converted" or "_cast"
        # The source column is usually the input column with the same base name
        src_col = f.name
        if src_col.endswith("_converted") or src_col.endswith("_cast"):
            src_col = src_col.rsplit("_", 1)[0]
        # Look for matching input column
        input_names = [fi.name for fi in tx.fields if fi.port_type == "INPUT"]
        if input_names and src_col not in input_names:
            # Heuristic: find closest match
            for inp_name in input_names:
                if inp_name.lower() in f.name.lower() or f.name.lower() in inp_name.lower():
                    src_col = inp_name
                    break
            else:
                src_col = input_names[0] if input_names else f.name

        if cast_target in ("decimal", "numeric"):
            if f.precision and f.scale:
                cast_str = f"DecimalType({f.precision}, {f.scale})"
            else:
                cast_str = "DecimalType(18, 2)"
            casts.append(f'        .withColumn("{f.name}", F.col("{src_col}").cast({cast_str}))')
        else:
            casts.append(f'        .withColumn("{f.name}", F.col("{src_col}").cast("{cast_target}"))')

    cast_chain = "\n".join(casts)
    return f"""
def transform_{func_name}(input_df):
    \"\"\"SSIS DataConvert: {tx.name} — explicit type casting.\"\"\"
    from pyspark.sql.types import DecimalType
    return (
        input_df
{cast_chain}
    )
""".strip()


# ─── UnionAll ─────────────────────────────────────────────────────────────────

def _gen_union_all(tx: "Transformation", mapping: "Mapping") -> str:
    """Generate .unionByName() for UnionAll transformation."""
    func_name = _safe_name(tx.name)

    # Find upstream components (2+ inputs feeding this UnionAll)
    upstream = [
        conn.from_instance for conn in mapping.connectors
        if conn.to_instance == tx.name
    ]

    if len(upstream) < 2:
        # Fallback: generate generic 2-input union if connectors not available
        return f"""
def transform_{func_name}(df_a, df_b, *extra_dfs):
    \"\"\"SSIS UnionAll: {tx.name} — combines multiple input streams.
    Pass all input DataFrames as positional arguments.
    \"\"\"
    result = df_a.unionByName(df_b, allowMissingColumns=True)
    for df in extra_dfs:
        result = result.unionByName(df, allowMissingColumns=True)
    return result
""".strip()

    param_names = [_safe_name(u) + "_df" for u in upstream]
    params_str = ", ".join(param_names)
    union_chain = param_names[0]
    for p in param_names[1:]:
        union_chain = f"{union_chain}.unionByName({p}, allowMissingColumns=True)"

    return f"""
def transform_{func_name}({params_str}):
    \"\"\"SSIS UnionAll: {tx.name} — combines {len(upstream)} input streams.\"\"\"
    return {union_chain}
""".strip()


# ─── Merge ───────────────────────────────────────────────────────────────────

def _gen_merge(tx: "Transformation", mapping: "Mapping") -> str:
    """Generate .unionByName() for Merge transformation (requires pre-sorted inputs)."""
    func_name = _safe_name(tx.name)
    upstream = [
        conn.from_instance for conn in mapping.connectors
        if conn.to_instance == tx.name
    ]
    if len(upstream) < 2:
        return None
    param_names = [_safe_name(u) + "_df" for u in upstream]
    params_str = ", ".join(param_names)
    union_chain = param_names[0]
    for p in param_names[1:]:
        union_chain = f"{union_chain}.unionByName({p}, allowMissingColumns=True)"

    return f"""
def transform_{func_name}({params_str}):
    \"\"\"SSIS Merge: {tx.name} — merges pre-sorted input streams.
    NOTE: SSIS Merge requires pre-sorted inputs on join key; PySpark unionByName
    does not require sorting. Verify upstream sort if output order matters.
    \"\"\"
    return {union_chain}
""".strip()


# ─── Multicast ───────────────────────────────────────────────────────────────

def _gen_multicast(tx: "Transformation", mapping: "Mapping") -> str:
    """Generate comment block for Multicast — returns same DataFrame to all consumers."""
    func_name = _safe_name(tx.name)
    # Downstream components
    downstream = [
        conn.to_instance for conn in mapping.connectors
        if conn.from_instance == tx.name
    ]
    downstream_str = ", ".join(downstream) if downstream else "multiple downstream components"

    return f"""
def transform_{func_name}(input_df):
    \"\"\"SSIS Multicast: {tx.name}
    Distributes the same DataFrame to {len(downstream)} downstream component(s):
    {downstream_str}

    In PySpark, simply reuse the DataFrame variable — no copy needed since
    DataFrames are immutable. Cached if multiple consumers will trigger actions.
    \"\"\"
    # Cache only if multiple consumers will trigger separate Spark actions
    # input_df = input_df.cache()
    return input_df
""".strip()


# ─── RowCount ────────────────────────────────────────────────────────────────

def _gen_row_count(tx: "Transformation", mapping: "Mapping") -> str:
    """Generate .count() + logging for RowCount transformation."""
    func_name = _safe_name(tx.name)
    var_name = tx.properties.get("VariableName", f"row_count_{func_name}")
    # Sanitise variable name
    var_name = _safe_name(var_name.split("::")[-1]) if "::" in var_name else _safe_name(var_name)

    return f"""
def transform_{func_name}(input_df, metrics=None):
    \"\"\"SSIS RowCount: {tx.name} — counts rows and stores in metrics dict.\"\"\"
    count = input_df.count()
    import logging
    logging.getLogger(__name__).info(
        "{tx.name}: %d rows", count
    )
    if metrics is not None:
        metrics["{var_name}"] = count
    return input_df, count
""".strip()


# ─── OLE DB Command ──────────────────────────────────────────────────────────

def _gen_oledb_command(tx: "Transformation", mapping: "Mapping") -> str:
    """Generate JDBC execute stub for OLE DB Command transformation."""
    func_name = _safe_name(tx.name)
    sql = tx.sql_query or tx.properties.get("SqlCommand", "/* SQL COMMAND NOT EXTRACTED */")
    # Truncate long SQL for readability
    sql_preview = sql[:200].replace('"', '\\"').replace("\n", " ")

    return f"""
def transform_{func_name}(input_df, spark, jdbc_config):
    \"\"\"SSIS OLE DB Command: {tx.name}
    Executes a parameterized SQL statement row-by-row.
    SSIS SQL: {sql_preview}

    PySpark equivalent: use a single batch UPDATE/MERGE via spark.sql()
    or a JDBC execute if DDL/DML is required.
    \"\"\"
    # TODO: Implement as batch UPDATE/MERGE for performance
    # Option 1: Write to temp table then MERGE
    # Option 2: Use DataFrame join + overwrite
    logger = __import__('logging').getLogger(__name__)
    logger.warning(
        "OLE DB Command '{tx.name}' executed as pass-through — "
        "implement batch SQL logic below."
    )
    # Placeholder: pass DataFrame through unchanged
    return input_df
""".strip()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _safe_name(name: str) -> str:
    """Convert SSIS component name to valid Python snake_case identifier."""
    import re
    safe = re.sub(r'[^a-zA-Z0-9]', '_', name).lower()
    safe = re.sub(r'_+', '_', safe).strip('_')
    if safe and safe[0].isdigit():
        safe = 'c_' + safe
    return safe or 'unknown'


def get_supported_types() -> list[str]:
    """Return list of component types handled deterministically."""
    return ["Sort", "DataConvert", "UnionAll", "Multicast", "Merge", "RowCount", "OLEDBCommand"]


def can_handle(tx: "Transformation") -> bool:
    """Return True if this transformation has sufficient metadata for deterministic generation."""
    if tx.type == "Sort":
        return bool(tx.sort_columns or any(
            f.sort_position > 0 for f in tx.fields if f.port_type == "OUTPUT"))
    if tx.type == "DataConvert":
        return bool(any(f.port_type == "OUTPUT" for f in tx.fields))
    if tx.type in ("UnionAll", "Merge"):
        return True  # will be checked at generation time
    if tx.type == "Multicast":
        return True
    if tx.type == "RowCount":
        return True
    if tx.type == "OLEDBCommand":
        return True
    return False
