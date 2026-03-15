"""
PySpark Code Generator.

Orchestrates the conversion of parsed SSIS packages
into PySpark code using LLM-based generation.

Produces Medallion Architecture output:
- Silver layer: Cleansed and transformed data
- Gold layer: Aggregated, business-ready data
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import re as _re_gen

from src.config import get_config
from src.llm import BaseLLMProvider
from src.logging import get_logger, LogContext
from src.parsers import Mapping, Transformation, Workflow

# Production enhancement modules (graceful fallback if unavailable)
try:
    from src.converters.deterministic_converter import try_deterministic, can_handle
    from src.generators.code_healer import apply_rule_patches, scan_for_warnings, build_healing_prompt
    _ENHANCEMENTS_AVAILABLE = True
except ImportError:
    _ENHANCEMENTS_AVAILABLE = False
    def try_deterministic(tx, mapping): return None
    def can_handle(tx): return False
    def apply_rule_patches(code): 
        class _R: 
            def __init__(self, c): self.code=c; self.patches_applied=[]; self.syntax_ok=True; self.syntax_error=None
        return _R(code)
    def scan_for_warnings(code, context=""): return []
    def build_healing_prompt(code, warnings, system_prompt=""): return ""

logger = get_logger(__name__)


@dataclass
class GenerationResult:
    """Result of PySpark generation."""
    success: bool
    files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    metadata: list[dict] = field(default_factory=list)
    # Conversion quality metrics
    coverage_score: float = 0.0   # 0.0–1.0 — fraction of components successfully generated
    todo_count: int = 0            # Number of # TODO comments in generated code
    stub_count: int = 0            # Number of NotImplementedError stubs in generated code


class PySparkGenerator:
    """
    Generates PySpark code from parsed SSIS packages.
    
    Strategy:
    1. Analyze workflow structure
    2. Classify mappings into Silver vs Gold
    3. Generate PySpark for each transformation via LLM
    4. Assemble per-mapping code via LLM
    5. Tailor into cohesive Silver and Gold notebooks via LLM
    6. Generate main orchestrator (Silver -> Gold)
    """
    
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        verbose: bool = False,
    ):
        self.llm = llm_provider
        self.verbose = verbose
        self.config = get_config()
        
        # Load system prompt
        self.system_prompt = self.llm.load_prompt("system.md")
        transformation_rules = self.llm.load_prompt("transformation_rules.md")
        if transformation_rules:
            self.system_prompt += f"\n\n{transformation_rules}"
    
    def generate(self, workflow: Workflow, output_dir: Path) -> GenerationResult:
        """
        Generate PySpark code for an entire workflow.
        
        Produces a Medallion architecture output:
        - Silver notebook: cleansed/transformed data
        - Gold notebook: aggregated/business-ready data
        - Main orchestrator: Silver -> Gold in sequence
        
        Args:
            workflow: Parsed SSIS package
            output_dir: Directory to write generated files
            
        Returns:
            GenerationResult with file paths and any errors
        """
        result = GenerationResult(success=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with LogContext(workflow=workflow.name):
            logger.info(
                "Starting code generation (Medallion: Silver + Gold)",
                mappings=len(workflow.mappings),
                transformations=len(workflow.transformations),
            )
            
            # Step 0: Generate Bronze ingestion notebook
            try:
                bronze_path = self._generate_bronze_notebook(workflow, output_dir)
                result.files.insert(0, str(bronze_path))
                logger.info("Generated Bronze notebook", path=str(bronze_path))
            except Exception as e:
                error_msg = f"Error generating Bronze notebook: {e}"
                result.errors.append(error_msg)
                result.success = False
                logger.error("Bronze generation failed", error=str(e))

            # Step 1: Pre-Analysis / Reverse Engineering [NEW]
            try:
                logger.info("Starting Stage 0: Automated Reverse Engineering Analysis", workflow=workflow.name)
                analysis_path = self._analyze_package(workflow)
                logger.info("Automated analysis complete", path=str(analysis_path))
            except Exception as e:
                logger.warning(f"Stage 0 Analysis failed: {e}. Proceeding without automated analysis.")

            # Step 2: Classify mappings into Silver vs Gold
            silver_mappings, gold_mappings = self._classify_mappings(workflow)
            logger.info(
                "Classified mappings",
                silver=len(silver_mappings),
                gold=len(gold_mappings),
            )

            # Step 2: Generate code for each mapping via LLM (per-transformation)
            silver_code_blocks = {}  # mapping_name -> assembled module code
            gold_code_blocks = {}    # mapping_name -> assembled module code

            for mapping in workflow.mappings:
                try:
                    module_code, mapping_metadata = self._generate_mapping_code(mapping)
                    result.metadata.extend(mapping_metadata)

                    if mapping in silver_mappings:
                        silver_code_blocks[mapping.name] = module_code
                    else:
                        gold_code_blocks[mapping.name] = module_code

                    # Post-generation code validation
                    code_warnings = self._validate_generated_code(module_code, mapping.name)
                    result.warnings.extend(code_warnings)
                    if code_warnings:
                        logger.warning("Code validation warnings", mapping=mapping.name, count=len(code_warnings))

                    logger.info("Generated mapping code", mapping=mapping.name)
                except Exception as e:
                    error_msg = f"Error generating {mapping.name}: {e}"
                    result.errors.append(error_msg)
                    logger.error("Generation failed", mapping=mapping.name, error=str(e))

                    # Retry once before giving up
                    try:
                        logger.info("Retrying mapping generation", mapping=mapping.name)
                        module_code, mapping_metadata = self._generate_mapping_code(mapping)
                        result.metadata.extend(mapping_metadata)

                        if mapping in silver_mappings:
                            silver_code_blocks[mapping.name] = module_code
                        else:
                            gold_code_blocks[mapping.name] = module_code

                        logger.info("Retry succeeded", mapping=mapping.name)
                        result.errors.pop()  # Remove the error since retry succeeded
                    except Exception as retry_e:
                        # Add a stub so the tailoring LLM knows this mapping exists
                        tx_names = [tx.name for tx in mapping.transformations]
                        stub_code = (
                            f"# ERROR: Code generation failed for mapping: {mapping.name}\n"
                            f"# Transformations: {', '.join(tx_names)}\n"
                            f"# Error: {retry_e}\n"
                            f"# TODO: Manually implement this mapping\n"
                            f"def run_{mapping.name.lower().replace(' ', '_')}(spark, input_df):\n"
                            f"    raise NotImplementedError('Code generation failed for {mapping.name}')\n"
                        )
                        if mapping in silver_mappings:
                            silver_code_blocks[mapping.name] = stub_code
                        else:
                            gold_code_blocks[mapping.name] = stub_code

                        result.success = False
                        logger.error("Retry also failed", mapping=mapping.name, error=str(retry_e))

            # Step 3: Tailor Silver notebook via LLM
            silver_manifest: dict = {}  # entity_name → silver table names (populated below)
            try:
                silver_path = self._tailor_silver_notebook(
                    workflow, silver_mappings, gold_mappings, silver_code_blocks, output_dir
                )
                result.files.append(str(silver_path))
                logger.info("Generated Silver notebook", path=str(silver_path))

                # Emit silver_manifest.json — extract saveAsTable targets from generated Silver code
                silver_manifest = self._extract_silver_manifest(silver_path, workflow)
                import json as _json
                manifest_path = output_dir / "silver_manifest.json"
                manifest_path.write_text(
                    _json.dumps(silver_manifest, indent=2), encoding="utf-8"
                )
                result.files.append(str(manifest_path))
                logger.info("Emitted silver_manifest.json", tables=list(silver_manifest.get("silver_tables", [])))
            except Exception as e:
                error_msg = f"Error generating Silver notebook: {e}"
                result.errors.append(error_msg)
                result.success = False
                logger.error("Silver generation failed", error=str(e))

            # Step 4: Tailor Gold notebook via LLM
            try:
                gold_path = self._tailor_gold_notebook(
                    workflow, gold_mappings, gold_code_blocks, output_dir,
                    silver_mappings=silver_mappings,
                    silver_manifest=silver_manifest,
                )
                result.files.append(str(gold_path))
                logger.info("Generated Gold notebook", path=str(gold_path))
            except Exception as e:
                error_msg = f"Error generating Gold notebook: {e}"
                result.errors.append(error_msg)
                result.success = False
                logger.error("Gold generation failed", error=str(e))

            # Step 5: Generate main orchestration script (Bronze -> Silver -> Gold)
            main_path = self._generate_main_script(workflow, output_dir)
            result.files.insert(0, str(main_path))

            # Step 6: Generate utilities module
            utils_path = self._generate_utils(output_dir)
            result.files.append(str(utils_path))

            # Step 7: Generate test scaffold
            try:
                test_path = self._generate_test_scaffold(workflow, output_dir)
                result.files.append(str(test_path))
                logger.info("Generated test scaffold", path=str(test_path))
            except Exception as e:
                error_msg = f"Error generating test scaffold: {e}"
                result.errors.append(error_msg)
                logger.warning("Test scaffold generation failed", error=str(e))

            # Step 8: Compute coverage metrics across all generated files
            result = self._compute_coverage_metrics(result, workflow, silver_code_blocks, gold_code_blocks)

            logger.info(
                "Generation complete (Medallion)",
                success=result.success,
                files=len(result.files),
                errors=len(result.errors),
                coverage_score=f"{result.coverage_score:.1%}",
                todo_count=result.todo_count,
                stub_count=result.stub_count,
            )

        return result
    
    def _analyze_package(self, workflow: Workflow) -> Path:
        """Call LLM to perform granular reverse engineering analysis of the package.
        
        Saves result to prompts/analysis/<workflow_name>_analysis.md
        """
        analysis_sys_prompt = self.llm.load_knowledge("../package_analysis.md") or "You are an SSIS reverse engineering specialist."
        context = self._serialize_workflow_for_analysis(workflow)
        
        prompt = f"""
## Task: Perform Technical Reverse Engineering
Analyze the structured representation of the SSIS package below and generate an extensive reverse engineering document per the instructions in your system prompt.

### Package Metadata:
{context}
"""
        response = self.llm.generate(prompt, analysis_sys_prompt)
        analysis_text = response.text
        
        analysis_dir = Path("prompts/analysis")
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        pkg_name = self._sanitize_filename(workflow.name)
        file_path = analysis_dir / f"{pkg_name}_analysis.md"
        file_path.write_text(analysis_text, encoding="utf-8")
        
        return file_path

    def _serialize_workflow_for_analysis(self, workflow: Workflow) -> str:
        """Create a structured text representation of the workflow for RE analysis."""
        lines = [f"# Workflow: {workflow.name}", f"Description: {workflow.description or 'N/A'}", "\n## Sources:"]
        for s in workflow.sources:
            lines.append(f"- {s.name} ({s.type}) | Connection: {s.connection_manager}")
            if hasattr(s, 'sql_query') and s.sql_query:
                lines.append(f"  SQL Query: {s.sql_query}")
        
        lines.append(f"\n## Targets:")
        for t in workflow.targets:
            lines.append(f"- {t.name} ({t.type}) | Connection: {t.connection_manager}")
            
        lines.append(f"\n## Data Flow Tasks (Mappings):")
        for m in workflow.mappings:
            lines.append(f"### Mapping: {m.name}")
            tx_order = m.get_execution_order() if hasattr(m, 'get_execution_order') else m.transformations
            for tx in tx_order:
                lines.append(f"- {tx.name} ({tx.type})")
                if tx.sql_query:
                    lines.append(f"  SQL: {tx.sql_query}")
                if tx.filter_condition:
                    lines.append(f"  Filter: {tx.filter_condition}")
                if tx.join_condition or tx.lookup_condition:
                    lines.append(f"  Condition: {tx.join_condition or tx.lookup_condition}")
                if tx.expression_code:
                    lines.append(f"  Expression: {tx.expression_code}")
                if tx.script_code:
                    lines.append("  Contains Script Logic (Needs careful RE)")
            
        return "\n".join(lines)

    def _load_package_analysis(self, workflow_name: str) -> str:
        """Load a package-specific analysis document if it exists.
        
        Looks for prompts/analysis/<workflow_name>_analysis.md
        """
        analysis_dir = Path("prompts/analysis")
        # Support both exact name and sanitized name
        names_to_try = [
            f"{workflow_name}_analysis.md",
            f"{self._sanitize_filename(workflow_name)}_analysis.md"
        ]
        
        for name in names_to_try:
            analysis_path = analysis_dir / name
            if analysis_path.exists():
                try:
                    analysis_text = analysis_path.read_text(encoding="utf-8")
                    logger.info("Loaded package-specific analysis", path=str(analysis_path))
                    return f"\n\n## Package-Specific Reverse Engineering Analysis:\n{analysis_text}"
                except Exception as e:
                    logger.warning(f"Error reading analysis file {analysis_path}: {e}")
        
        return ""

    # ─── Post-Generation Code Validator ───────────────────────────────────────

    def _validate_generated_code(self, code: str, mapping_name: str) -> list[str]:
        """Check generated code for invalid PySpark patterns and syntax errors.

        v2.0: Delegates to the self-healer scanner for comprehensive checks.
        Returns a list of warning strings surfaced in GenerationResult.warnings.
        """
        if _ENHANCEMENTS_AVAILABLE:
            return scan_for_warnings(code, mapping_name)

        # Fallback if enhancements not available
        warnings_found = []
        forbidden_calls = [
            ("F.right(",    "Use F.expr(\"right(col, n)\") — F.right() does not exist"),
            ("F.left(",     "Use F.substring(col, 1, n) — F.left() does not exist"),
            ("F.replace(",  "Use F.regexp_replace() — F.replace() does not exist"),
            ("F.isnull(",   "Use col.isNull() — F.isnull() does not exist"),
            (".collect()",  "Avoid .collect() on large DataFrames"),
        ]
        for bad_call, suggestion in forbidden_calls:
            if bad_call in code:
                warnings_found.append(f"[{mapping_name}] {suggestion}")
        if '.option("query",' in code and '.option("dbtable",' in code:
            warnings_found.append(f"[{mapping_name}] JDBC conflict: query + dbtable both present")
        try:
            compile(code, f"<{mapping_name}>", "exec")
        except SyntaxError as syn_err:
            warnings_found.append(
                f"[{mapping_name}] SyntaxError at line {syn_err.lineno}: {syn_err.msg}")
        return warnings_found

    # ─── Silver Manifest ───────────────────────────────────────────────────────

    def _extract_silver_manifest(self, silver_path: Path, workflow: Workflow) -> dict:
        """Parse the generated Silver notebook to extract saveAsTable targets.

        Returns a dict with 'silver_tables' (list of table names WITHOUT schema prefix)
        that the Gold layer can use as a reliable manifest.
        """
        import re as _re
        silver_tables: list[str] = []
        try:
            silver_code = silver_path.read_text(encoding="utf-8")
            # Match saveAsTable("silver.entity") or saveAsTable('silver.entity')
            matches = _re.findall(
                r'saveAsTable\s*\(\s*["\']silver\.([a-zA-Z0-9_]+)["\']',
                silver_code,
            )
            silver_tables = list(dict.fromkeys(matches))  # deduplicate, preserve order
        except Exception as e:
            logger.warning("Could not extract silver manifest from Silver notebook", error=str(e))

        return {
            "workflow": workflow.name,
            "silver_tables": silver_tables,
            "silver_table_fqns": [f"silver.{t}" for t in silver_tables],
        }

    # ─── Coverage Metrics ──────────────────────────────────────────────────────

    def _compute_coverage_metrics(
        self,
        result: GenerationResult,
        workflow: Workflow,
        silver_code_blocks: dict,
        gold_code_blocks: dict,
    ) -> GenerationResult:
        """Compute conversion quality metrics for the GenerationResult.

        Sets:
          - coverage_score: fraction of mappings with non-stub generated code
          - todo_count: number of '# TODO' lines in all generated code
          - stub_count: number of NotImplementedError stubs
        """
        all_code = "\n".join(list(silver_code_blocks.values()) + list(gold_code_blocks.values()))
        total_mappings = len(workflow.mappings)

        # Stub detection — stub code contains NotImplementedError
        stub_mappings = sum(1 for code in (list(silver_code_blocks.values()) + list(gold_code_blocks.values()))
                            if "NotImplementedError" in code)

        result.coverage_score = ((total_mappings - stub_mappings) / total_mappings) if total_mappings > 0 else 0.0
        result.todo_count = all_code.count("# TODO")
        result.stub_count = all_code.count("NotImplementedError")

        logger.info(
            "Coverage metrics",
            coverage=f"{result.coverage_score:.1%}",
            todos=result.todo_count,
            stubs=result.stub_count,
        )
        return result

    # ─── Test Scaffold ────────────────────────────────────────────

    def _generate_test_scaffold(
        self,
        workflow: Workflow,
        output_dir: Path,
    ) -> Path:
        """Generate a pytest test scaffold for all Silver and Gold transform functions."""
        silver_mappings, gold_mappings = self._classify_mappings(workflow)
        
        # Load knowledge for test style consistency
        pyspark_patterns = self.llm.load_knowledge("pyspark_patterns.md")
        
        prompt = self._build_test_scaffold_prompt(
            workflow, silver_mappings, gold_mappings,
            pyspark_patterns=pyspark_patterns
        )
        response = self.llm.generate(prompt, self.system_prompt)
        code = self.llm.extract_code(response.text)
        pkg_name = self._sanitize_filename(workflow.name)
        file_path = output_dir / f"test_{pkg_name}.py"
        file_path.write_text(code, encoding="utf-8")
        return file_path

    def _build_test_scaffold_prompt(
        self,
        workflow: Workflow,
        silver_mappings: list,
        gold_mappings: list,
        pyspark_patterns: str = None,
    ) -> str:
        """Build a fully dynamic pytest test scaffold prompt.

        All column names, table names, primary-key candidates, filter/join/group-by
        conditions, and transformation types are derived at Python level from the
        parsed Mapping/Transformation objects.  Nothing is project-specific.
        """
        import json as _json

        pkg_name = self._sanitize_filename(workflow.name)

        # ── helpers ───────────────────────────────────────────────────────────
        def _pk_candidates(fields) -> list[str]:
            """Heuristic PK detection: fields named *_id, *_code, *_key, *_no, or first field."""
            pk_suffixes = ("_id", "_key", "_code", "_no", "_num", "_nbr", "_sk", "_bk")
            pks = [f.name for f in fields if any(f.name.lower().endswith(s) for s in pk_suffixes)]
            return pks or ([fields[0].name] if fields else [])

        def _sample_value(datatype: str) -> str:
            """Return a safe, typed sample value string for a SSIS datatype."""
            dt = (datatype or "").lower()
            if dt in ("i4", "i8", "i2", "i1"):      return "1"
            if dt in ("r4", "r8", "numeric", "cy"):  return "1.0"
            if dt in ("bool",):                       return "True"
            if dt in ("dbdate", "dbtime", "dbtimestamp", "dbtimestamp2"):
                return '"2024-01-01"'
            return '"test_value"'

        def _mapping_meta(m) -> dict:
            """Extract a compact metadata dict from one Mapping object."""
            # Collect all input field names from Source objects
            src_cols = []
            for s in m.sources:
                for f in s.fields:
                    if f.name and f.name not in src_cols:
                        src_cols.append({"name": f.name, "type": f.datatype,
                                         "sample": _sample_value(f.datatype)})

            # Collect all output field names from Target objects
            tgt_cols = []
            for t in m.targets:
                for f in t.fields:
                    if f.name and f.name not in [c["name"] for c in tgt_cols]:
                        tgt_cols.append({"name": f.name, "type": f.datatype})

            # If targets have no explicit fields, fall back to OUTPUT-port fields
            if not tgt_cols:
                for tx in m.transformations:
                    for f in tx.fields:
                        if f.port_type == "OUTPUT" and f.name:
                            if f.name not in [c["name"] for c in tgt_cols]:
                                tgt_cols.append({"name": f.name, "type": f.datatype})

            # Gather transformation metadata
            transforms = []
            for tx in m.transformations:
                t_meta = {
                    "name": tx.name,
                    "type": tx.type,
                    "filter_condition":  tx.filter_condition or None,
                    "join_condition":    tx.join_condition   or None,
                    "lookup_condition":  tx.lookup_condition or None,
                    "group_by":          tx.group_by         or [],
                    "output_columns":    [
                        {"name": f.name, "type": f.datatype, "expression": f.expression}
                        for f in tx.fields if f.port_type == "OUTPUT" and f.name
                    ][:15],  # cap at 15 to keep prompt compact
                }
                transforms.append(t_meta)

            # Source / target table name heuristic
            src_names = [s.name for s in m.sources] or ["bronze_source"]
            tgt_names = [t.name for t in m.targets] or ["silver_target"]

            all_src_col_names = [c["name"] for c in src_cols]
            pks = _pk_candidates([type("F", (), {"name": c["name"]})() for c in src_cols])

            # Detect lookup transformations for join-no-match test
            lookup_txs = [tx for tx in m.transformations if tx.type in ("Lookup", "lookup")]
            lookup_keys = []
            for lt in lookup_txs:
                if lt.lookup_condition:
                    # extract column names from condition like "a.col = b.col"
                    import re as _re
                    lookup_keys.extend(_re.findall(r'\b\w+\b', lt.lookup_condition))

            # Detect parallel-capable transforms (same type, no dependency)
            parallel_pairs = []
            tx_names = [tx.name for tx in m.transformations if tx.type not in ("Lookup",)]
            if len(tx_names) >= 2:
                parallel_pairs = tx_names[:2]

            return {
                "mapping_name":     m.name,
                "function_name":    f"transform_{self._sanitize_filename(m.name)}",
                "source_tables":    src_names,
                "target_tables":    tgt_names,
                "source_columns":   src_cols[:20],      # cap for prompt size
                "target_columns":   tgt_cols[:20],
                "primary_keys":     pks,
                "transformations":  transforms,
                "has_lookup":       bool(lookup_txs),
                "lookup_keys":      list(set(lookup_keys))[:5],
                "parallel_pairs":   parallel_pairs,
            }

        silver_meta = [_mapping_meta(m) for m in silver_mappings]
        gold_meta   = [_mapping_meta(m) for m in gold_mappings]

        silver_meta_json = _json.dumps(silver_meta, indent=2)
        gold_meta_json   = _json.dumps(gold_meta,   indent=2)

        # Derive data-quality checks from all target tables
        dq_table_checks = []
        for sm in silver_meta:
            for tbl in sm["target_tables"]:
                for pk in sm["primary_keys"]:
                    dq_table_checks.append({"silver_table": f"silver.{tbl}", "pk": pk})
        dq_json = _json.dumps(dq_table_checks[:10], indent=2)

        # Derive referential integrity checks: find FK columns in one silver table
        # that match a PK column in another (column name overlap heuristic)
        pk_set = {}
        for sm in silver_meta:
            for tbl in sm["target_tables"]:
                for pk in sm["primary_keys"]:
                    pk_set[pk] = f"silver.{tbl}"
        ri_checks = []
        for sm in silver_meta:
            for tbl in sm["target_tables"]:
                for col in sm["source_columns"]:
                    if col["name"] in pk_set and pk_set[col["name"]] != f"silver.{tbl}":
                        ri_checks.append({
                            "child_table":  f"silver.{tbl}",
                            "fk_column":    col["name"],
                            "parent_table": pk_set[col["name"]],
                            "pk_column":    col["name"],
                        })
        ri_json = _json.dumps(ri_checks[:5], indent=2)

        knowledge_section = ""
        if pyspark_patterns:
            knowledge_section = f"\n## PySpark Patterns Reference:\n{pyspark_patterns}\n"

        return f"""
## Task: Generate Comprehensive pytest Test Suite

Generate a **full validation test suite** for the PySpark Medallion pipeline for `{workflow.name}`.
The file must contain ALL five sections below — do NOT omit any section.
**CRITICAL**: Use ONLY the column names, table names, function names, and conditions from the
metadata manifests below.  Do NOT invent column/table names that are not listed there.

### Modules under test:
- `bronze_{pkg_name}` — Bronze ingestion
- `silver_{pkg_name}` — Silver transforms
- `gold_{pkg_name}` — Gold aggregations

{knowledge_section}
---

## Metadata Manifests

### Silver Mapping Metadata (use these for all Silver tests):
```json
{silver_meta_json}
```

### Gold Mapping Metadata (use these for all Gold tests):
```json
{gold_meta_json}
```

### Data-Quality Table / PK Checks (derived from targets):
```json
{dq_json}
```

### Referential Integrity Checks (FK → PK heuristic):
```json
{ri_json}
```

---

## Section 1 — File Header & Fixtures

Generate this header block exactly once at the top of the file:

```python
\"\"\"
Auto-generated comprehensive test suite for {workflow.name} Medallion Pipeline.
Sections:
  8.1 Unit Tests        — per-mapping transform logic and field-level rules
  8.2 Integration Tests — Silver-to-Gold end-to-end pipeline
  8.3 Data Quality      — no data loss, referential integrity, null PKs
  8.4 Performance       — broadcast join plan, SLA timing, partition skew
\"\"\"
import json
import pytest
import time
import traceback
import concurrent.futures
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F


# ── Session-scoped SparkSession ───────────────────────────────────────────────
@pytest.fixture(scope="session")
def spark():
    return SparkSession.builder \\
        .master("local[2]") \\
        .appName("test_{pkg_name}") \\
        .config("spark.sql.shuffle.partitions", "2") \\
        .getOrCreate()
```

---

## Section 2 — 8.1 Unit Tests (Transform Logic)

Using the **Silver Mapping Metadata** manifest above, generate ONE test function per mapping.
Follow this generic pattern (substitute all `<placeholders>` from the manifest — NEVER invent names):

```python
# ── 8.1 Unit Tests ──────────────────────────────────────────────────────────
def test_transform_<mapping.function_name>(spark):
    \"\"\"Smoke test: <mapping.mapping_name> — verifies transform produces expected output schema.
    SSIS source: <mapping.source_tables>  →  target: <mapping.target_tables>
    \"\"\"
    # Minimal 3-row DataFrame; column names and sample values from manifest source_columns
    sample = spark.createDataFrame(
        [(<mapping.source_columns[*].sample joined as tuple> for each of 1-3 rows)],
        [<mapping.source_columns[*].name as list>],
    )
    from silver_{pkg_name} import <mapping.function_name>
    result = <mapping.function_name>(sample)

    assert isinstance(result, DataFrame), "<mapping.function_name> must return a DataFrame"
    for col_name in [<mapping.target_columns[*].name>]:
        assert col_name in result.columns, f"Expected column {{col_name}} missing"
    assert result.count() >= 1, "Row count must be >= 1 after transform"


def test_transform_<mapping.function_name>_null_input(spark):
    \"\"\"Null-handling: all nullable inputs return a result without raising exceptions.\"\"\"
    null_row = spark.createDataFrame(
        [(None,) * <len(mapping.source_columns)>],
        [<mapping.source_columns[*].name>],
    )
    from silver_{pkg_name} import <mapping.function_name>
    result = <mapping.function_name>(null_row)  # must not raise
    assert result is not None


def test_transform_<mapping.function_name>_filter_condition(spark):
    \"\"\"Filter condition: only rows satisfying the SSIS condition pass through.
    filter_condition from manifest: <mapping.transformations[?].filter_condition>
    \"\"\"
    # Only generate this test if filter_condition is not null in the manifest.
    # Create 2 rows: one that satisfies the filter, one that does not.
    # Assert result.count() == 1.
    pytest.skip("Filter condition test — implement using actual filter column from manifest")
```

### DerivedColumn / Expression tests

For every mapping transformation of type `DerivedColumn` in the manifest, where
`output_columns[].expression` is non-empty:
- Create sample input with the input columns the expression depends on
- Run the transform
- Assert the output column value matches the expected expression result

### Gold aggregation tests

For EVERY entry in the **Gold Mapping Metadata** manifest, generate a `test_aggregate_<mapping.function_name>`:
- Use `source_columns` from that gold mapping's manifest for input schema
- Group by columns listed in `transformations[].group_by`
- Assert all output columns from `target_columns` exist in the result
- Assert numeric aggregations (SUM, COUNT, AVG) are >= 0 where applicable

---

## Section 5 — Data Quality Validator Class

Generate a reusable `DataQualityValidator` class with the following generic check methods.
Use ONLY table names and column names from the metadata manifests above.

```python
class DataQualityValidator:
    '''Reusable framework for data quality checks.'''

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.results = []

    def check_null_values(self, df: DataFrame, columns: list, table_name: str):
        '''Check for NULL values in required columns.'''
        for col_name in columns:
            null_count = df.filter(F.col(col_name).isNull()).count()
            total = df.count()
            self.results.append({{
                "check": "null_values", "table": table_name, "column": col_name,
                "null_count": null_count, "total": total,
                "status": "FAIL" if null_count > 0 else "PASS"
            }})

    def check_duplicates(self, df: DataFrame, key_columns: list, table_name: str):
        '''Check for duplicate records on key columns.'''
        total = df.count()
        distinct = df.select(key_columns).distinct().count()
        self.results.append({{
            "check": "duplicates", "table": table_name,
            "duplicate_count": total - distinct, "status": "FAIL" if total != distinct else "PASS"
        }})

    def check_referential_integrity(self, child_df, parent_df, child_key, parent_key, name):
        '''Check FK→PK relationships using left_anti join.'''
        orphaned = child_df.join(parent_df, child_df[child_key] == parent_df[parent_key], "left_anti")
        count = orphaned.count()
        self.results.append({{
            "check": "referential_integrity", "relationship": name,
            "orphaned_count": count, "status": "FAIL" if count > 0 else "PASS"
        }})

    def check_value_ranges(self, df, column, min_val, max_val, table_name):
        '''Check if values are within expected range.'''
        violations = df.filter((F.col(column) < min_val) | (F.col(column) > max_val)).count()
        self.results.append({{
            "check": "value_range", "table": table_name, "column": column,
            "violations": violations, "status": "FAIL" if violations > 0 else "PASS"
        }})

    def generate_report(self) -> list:
        return self.results
```

Then generate concrete test functions that instantiate `DataQualityValidator` and call
the check methods using actual table names and PK columns from the DQ and RI manifests:

```python
def test_data_quality_silver_tables(spark):
    '''Run data quality checks on all Silver output tables.'''
    validator = DataQualityValidator(spark)
    # For each entry in the DQ manifest:
    #   validator.check_null_values(spark.table(<table>), <pk_columns>, <table>)
    #   validator.check_duplicates(spark.table(<table>), <pk_columns>, <table>)
    report = validator.generate_report()
    failed = [r for r in report if r["status"] == "FAIL"]
    assert len(failed) == 0, f"Data quality failures: {{failed}}"

def test_referential_integrity(spark):
    '''Check FK→PK relationships between Silver tables.'''
    validator = DataQualityValidator(spark)
    # For each entry in the RI manifest:
    #   validator.check_referential_integrity(...)
    report = validator.generate_report()
    failed = [r for r in report if r["status"] == "FAIL"]
    assert len(failed) == 0, f"RI failures: {{failed}}"
```


"""

    # ─── Mapping Classification ──────────────────────────────────

    def _classify_mappings(self, workflow: Workflow) -> tuple[list, list]:
        """
        Classify mappings into Silver (transform/cleanse) vs Gold (aggregate/business).
        - **GOLD**: Data Flow Tasks with 'Aggregate' components OR names containing 'Aggregation'/'Summary'.
        - **SILVER**: All other cleansing and transformation tasks.
        """
        silver = []
        gold = []
        
        gold_types = {"Aggregate"}
        gold_keywords = {"aggregation", "summary", "agg_"}
        
        for mapping in workflow.mappings:
            tx_types = {tx.type for tx in mapping.transformations}
            name_lower = mapping.name.lower()
            
            # Heuristic: if it has an Aggregate transform OR "Aggregation"/"Summary" in the name
            if (tx_types & gold_types) or any(k in name_lower for k in gold_keywords):
                gold.append(mapping)
            else:
                silver.append(mapping)
        
        return silver, gold

    def _build_control_flow_context(self, workflow: 'Workflow', silver_mappings: list = None, gold_mappings: list = None) -> str:
        """Build a human-readable control flow hierarchy for LLM prompts.

        Serializes the SSIS control flow (Sequence Containers, Foreach Loops,
        Execute SQL Tasks, Data Flow Tasks) into a structured text block that
        gives the LLM full visibility of the package's execution structure.
        """
        if not workflow.control_flow_tasks and not workflow.precedence_constraints:
            return ""

        lines = ["### SSIS Control Flow Hierarchy:"]

        # Group tasks by parent path to build tree
        by_parent: dict[str, list] = {}
        for task in workflow.control_flow_tasks:
            parent = task.parent_path or "(root)"
            by_parent.setdefault(parent, []).append(task)

        # Also add DFTs by hierarchy
        dft_names = {dft.name for dft in workflow.data_flow_tasks}

        def _render_tree(parent_key: str, indent: int = 0):
            children = by_parent.get(parent_key, [])
            for task in children:
                prefix = "  " * indent + "├── " if indent > 0 else "- "
                task_type_short = task.properties.get("ContainerType", task.task_type.split(",")[0].split(".")[-1])
                disabled_str = " [DISABLED]" if task.disabled else ""
                
                # Tag task as [SILVER] or [GOLD] if it's a DFT mapping
                tag = ""
                if silver_mappings and any(m.name == task.name for m in silver_mappings):
                    tag = " [SILVER]"
                elif gold_mappings and any(m.name == task.name for m in gold_mappings):
                    tag = " [GOLD]"
                
                lines.append(f"{prefix}{task.name} ({task_type_short}){disabled_str}{tag}")

                # Add SQL statement for Execute SQL tasks
                sql = task.properties.get("SqlStatementSource", "")
                if sql:
                    sql_preview = sql.strip().replace("\n", " ")[:120]
                    lines.append(f"{'  ' * (indent + 1)}   SQL: {sql_preview}")

                # Add result set info
                result_type = task.properties.get("ResultSetType", "")
                if result_type:
                    lines.append(f"{'  ' * (indent + 1)}   ResultSet: {result_type}")

                # Add Foreach Loop details
                enum_type = task.properties.get("EnumeratorType", "")
                if enum_type:
                    lines.append(f"{'  ' * (indent + 1)}   Enumerator: {enum_type}")

                var_mappings = task.properties.get("VariableMappings", [])
                if var_mappings:
                    mapping_strs = [f"{vm['variable']}[{vm['index']}]" for vm in var_mappings]
                    lines.append(f"{'  ' * (indent + 1)}   VarMappings: {', '.join(mapping_strs)}")

                # If this is a Data Flow Task (in control flow list), note component count
                if task.name in dft_names:
                    dft = next((d for d in workflow.data_flow_tasks if d.name == task.name), None)
                    if dft:
                        lines.append(f"{'  ' * (indent + 1)}   Components: {len(dft.components)} ({len(dft.sources)} src, {len(dft.destinations)} dest, {len(dft.transformations)} transforms)")

                # Recurse into children
                child_key = f"{parent_key}\\{task.name}" if parent_key != "(root)" else task.name
                _render_tree(child_key, indent + 1)

        _render_tree("(root)")

        # Add top-level DFTs that weren't in control flow (edge case)
        top_level_dfts = [dft for dft in workflow.data_flow_tasks
                         if not any(t.name == dft.name for t in workflow.control_flow_tasks)]
        for dft in top_level_dfts:
            lines.append(f"- {dft.name} (DataFlowTask)")
            lines.append(f"    Components: {len(dft.components)} ({len(dft.sources)} src, {len(dft.destinations)} dest)")

        # Add precedence constraints
        if workflow.precedence_constraints:
            lines.append("\n### Execution Order (Precedence Constraints):")
            value_map = {0: "Success", 1: "Failure", 2: "Completion"}
            for pc in workflow.precedence_constraints:
                constraint_type = value_map.get(pc.value, "Unknown")
                expr_str = f" [expr: {pc.expression}]" if pc.expression else ""
                lines.append(f"  {pc.from_task} --({constraint_type})--> {pc.to_task}{expr_str}")

        return "\n".join(lines)

    
    # ─── Per-Mapping Code Generation (LLM per transformation) ────
    
    def _generate_mapping_code(self, mapping: Mapping) -> tuple[str, list[dict]]:
        """
        Generate PySpark code for a single mapping (returns code string, not file).
        
        This calls the LLM for each transformation, then assembles them.
        """
        metadata = []
        with LogContext(mapping=mapping.name):
            logger.debug("Generating mapping", transformations=len(mapping.transformations))
            
            # Get execution order
            ordered_transformations = mapping.get_execution_order()
            
            # Generate code for each transformation via LLM
            code_blocks = []
            for tx in ordered_transformations:
                if self.verbose:
                    logger.debug("Generating transformation", type=tx.type, name=tx.name)
                
                code, notes = self._generate_transformation(tx, mapping)
                code_blocks.append(code)
                
                if notes:
                    metadata.append({
                        "mapping": mapping.name,
                        "transformation": tx.name,
                        "type": tx.type,
                        "notes": notes
                    })
            
            # Assemble into module using LLM
            module_code = self._assemble_module(mapping.name, code_blocks, mapping)
            
            return module_code, metadata
    
    def _generate_transformation(self, tx: Transformation, mapping: Mapping) -> tuple[str, str]:
        """Generate PySpark code for a single transformation.
        
        v2.0: Tries deterministic generation first (Sort, DataConvert, UnionAll, etc.)
        then falls back to LLM. Applies self-healing patches after LLM generation.
        """
        # ── STEP 1: Deterministic code generation (no LLM) ─────────────────
        if _ENHANCEMENTS_AVAILABLE and can_handle(tx):
            det_code = try_deterministic(tx, mapping)
            if det_code:
                logger.debug("Deterministic generation", type=tx.type, name=tx.name)
                # Still run through healer for safety
                healed = apply_rule_patches(det_code)
                return healed.code, f"Deterministic generation ({tx.type})"

        # ── STEP 2: Load knowledge ──────────────────────────────────────────
        template = self.llm.load_template(tx.type)
        type_mappings = self.llm.load_knowledge("type_mappings.yaml")
        pyspark_patterns = self.llm.load_knowledge("pyspark_patterns.md")
        variable_rules = self.llm.load_knowledge("variable_rules.md")
        connection_manager_rules = self.llm.load_knowledge("connection_manager_rules.md")
        silver_patterns = self.llm.load_knowledge("silver_layer_patterns.md")
        transformation_rules = self.llm.load_knowledge("../transformation_rules.md")
        ssis_xml_ref = self.llm.load_knowledge("ssis_xml_reference.md")
        env_references = self.llm.load_knowledge("ssis_reference/environment_references.md")
        project_params = self.llm.load_knowledge("ssis_reference/project_parameters.md")
        fabric_patterns = self.llm.load_knowledge("microsoft_fabric_patterns.md")

        context = self._build_context(tx, mapping)
        package_analysis = self._load_package_analysis(mapping.workflow_name)

        # ── STEP 3: Build enriched prompt with all extracted metadata ───────
        prompt = self._assemble_prompt(
            transformation=tx,
            template=template,
            type_mappings=type_mappings,
            pyspark_patterns=pyspark_patterns,
            context=context,
            variable_rules=variable_rules,
            connection_manager_rules=connection_manager_rules,
            package_analysis=package_analysis,
        )
        if silver_patterns:
            prompt += f"\n\n## Silver Layer Patterns Reference:\n{silver_patterns}"
        if transformation_rules:
            prompt += f"\n\n## Transformation Rules:\n{transformation_rules}"
        if ssis_xml_ref:
            prompt += f"\n\n## SSIS XML Reference:\n{ssis_xml_ref}"
        if env_references:
            prompt += f"\n\n## Environment References:\n{env_references}"
        if project_params:
            prompt += f"\n\n## Project Parameters:\n{project_params}"
        if fabric_patterns:
            prompt += f"\n\n## Microsoft Fabric Architecture:\n{fabric_patterns}"

        # ── STEP 4: LLM generation ──────────────────────────────────────────
        response = self.llm.generate(prompt, self.system_prompt)
        code = self.llm.extract_code(response.text)
        notes = _re_gen.sub(r'```(?:python)?.*?```', '', response.text, flags=_re_gen.DOTALL).strip()

        # ── STEP 5: Self-healing patches ────────────────────────────────────
        if _ENHANCEMENTS_AVAILABLE:
            healed = apply_rule_patches(code)
            if healed.patches_applied:
                logger.info(
                    "Self-healer applied patches",
                    transformation=tx.name,
                    patches=healed.patches_applied,
                )
                code = healed.code

            # ── STEP 6: LLM-assisted healing for syntax errors ───────────────
            if not healed.syntax_ok:
                logger.warning("Syntax error in generated code — attempting LLM heal",
                               transformation=tx.name, error=healed.syntax_error)
                remaining_warnings = scan_for_warnings(code, tx.name)
                heal_prompt = build_healing_prompt(code, remaining_warnings, self.system_prompt)
                try:
                    heal_response = self.llm.generate(heal_prompt, self.system_prompt)
                    healed_code = self.llm.extract_code(heal_response.text)
                    # Verify the healed code compiles
                    compile(healed_code, f"<healed:{tx.name}>", "exec")
                    code = healed_code
                    logger.info("LLM healing succeeded", transformation=tx.name)
                except Exception as heal_err:
                    logger.warning("LLM healing failed", transformation=tx.name, error=str(heal_err))
                    # Keep best version we have (rule-patched but with syntax issue)

        return code, notes
    
    def _assemble_prompt(
        self,
        transformation: Transformation,
        template: str,
        type_mappings: str,
        pyspark_patterns: str,
        context: Optional[str],
        variable_rules: Optional[str] = None,
        connection_manager_rules: Optional[str] = None,
        package_analysis: Optional[str] = None,
    ) -> str:
        """Assemble the complete prompt for code generation."""
        parts = []
        
        if context:
            parts.append(f"## Context\n{context}")
        
        if package_analysis:
            parts.append(f"## Package-Wide Business Logic & Domain Analysis (Source of Truth)\n{package_analysis}")
        
        if type_mappings:
            parts.append(f"## Data Type Mappings\n```yaml\n{type_mappings}\n```")
        
        if pyspark_patterns:
            parts.append(f"## PySpark Patterns Reference\n{pyspark_patterns}")
        
        if variable_rules:
            parts.append(f"## Variable Conversion Rules\n{variable_rules}")
        
        if connection_manager_rules:
            parts.append(f"## Connection Manager Rules\n{connection_manager_rules}")
        
        if template:
            parts.append(f"## Transformation Template\n{template}")
        
        # Add the transformation details
        parts.append(f"""
## Transformation to Convert

**Type:** {transformation.type}
**Name:** {transformation.name}
**Description:** {transformation.description or 'N/A'}

### Fields/Ports:
```json
{json.dumps([{k:v for k,v in vars(f).items() if k not in ("sort_position","sort_descending","cast_type")} for f in transformation.fields], indent=2)}
```

### Properties:
```json
{json.dumps(transformation.properties, indent=2)}
```

### Type-Specific Info (EXTRACTED FROM DTSX — USE EXACTLY):
- SQL Query: {transformation.sql_query or 'N/A'}
- Group By Columns (Aggregate): {transformation.group_by or []}
- Filter Condition: {transformation.filter_condition or 'N/A'}
{("- Join Type: " + str(getattr(transformation, "join_type", "inner")) + "\n- Num Key Columns: " + str(getattr(transformation, "num_key_columns", 0)) + "\n- Join Condition: " + (transformation.join_condition or "N/A")) if transformation.type in ("MergeJoin","Merge") else ""}
{("- Lookup Condition: " + (transformation.lookup_condition or "N/A") + "\n- Lookup No-Match Behavior: " + str(getattr(transformation, "lookup_no_match_behavior", "redirect")) + "\n- Lookup Reference SQL: " + (getattr(transformation, "lookup_reference_sql", "") or "N/A")) if transformation.type == "Lookup" else ""}
{("- Sort Columns: " + json.dumps(getattr(transformation, "sort_columns", []), indent=2)) if transformation.type == "Sort" else ""}
{("- Output Branch Conditions:\n" + json.dumps(transformation.output_conditions, indent=2)) if transformation.type == "ConditionalSplit" else ""}
{("- Script Language: " + str(getattr(transformation, "script_language", "")) + "\n- Read-Only Variables: " + str(getattr(transformation, "script_read_only_vars", "")) + "\n- Script Code:\n```csharp\n" + (transformation.script_code or "# not available") + "\n```") if transformation.type == "Script" else ""}

## Task
Generate clean, production-ready PySpark code that implements this transformation.

**STRICT FIDELITY RULES (MANDATORY):**
- **METADATA IS GROUND TRUTH**: Use ONLY the column names, data types, and properties provided in the JSON metadata above. 
- **NO HALLUCINATIONS**: Do NOT add "helpful" columns (like year, month, or batch IDs) if they are not explicitly present in the transformation's input or output ports.
- **NAMING**: Match the SSIS destination mapping EXACTLY. Rename columns if the destination expects a different name than the source.
- **GROUP BY**: For aggregations, use ONLY the columns marked for "Group By" in the metadata. NEVER add extra grouping keys.
- **EXPRESSIONS**: Replicate SSIS expression logic exactly. If a string comparison is used (e.g., `[code] == "1"`), do NOT assume it's an integer comparison.
- **AGGREGATE OUTPUT NAMES (CRITICAL)**: For Aggregate transformations, each OUTPUT field in the metadata has `name`, `agg_type`, and `src_col`.
  Use EXACTLY the `name` field as the `.alias()` value and map `agg_type` to PySpark:
  - `SUM` → `F.sum("src_col").alias("name")`
  - `AVG` → `F.avg("src_col").alias("name")`
  - `MIN` → `F.min("src_col").alias("name")`
  - `MAX` → `F.max("src_col").alias("name")`
  - `COUNT` → `F.count("src_col").alias("name")`
  - `COUNT_DISTINCT` → `F.countDistinct("src_col").alias("name")`
  - `GROUP_BY` → include in `.groupBy()`
  Do NOT invent column names like `total_*`, `min_*`, `max_*` — use the exact `name` from the metadata.

- Include comments explaining the business logic.
- Use DataFrame API (not RDD).
- Include type hints.
- Chaining: Return the DataFrame so it can be used in subsequent transformations.

## Conversion Notes Requirement
In addition to the code, please provide:
1. **Optimizations Applied**: What improvements did you make over the original SSIS logic? (e.g. vectorization, performance).
2. **Review Notes**: Are there any parts of the logic you are unsure about? What should a developer check specifically?
3. **Complexity Note**: Rate the conversion complexity (Low/Medium/High).

Provide these notes BEFORE the code block.
""")
        
        return "\n\n".join(parts)
    
    def _build_context(self, tx: Transformation, mapping: Mapping) -> str:
        """Build context string describing upstream transformations."""
        upstream = []
        for conn in mapping.connectors:
            if conn.to_instance == tx.name:
                src_info = conn.from_instance
                if conn.from_field:
                    src_info += f" (output: {conn.from_field})"
                upstream.append(src_info)
        
        if upstream:
            return f"Upstream transformations: {', '.join(sorted(upstream))}"
        return ""
    
    def _assemble_module(
        self,
        mapping_name: str,
        code_blocks: list[str],
        mapping: Mapping,
        package_analysis: Optional[str] = None,
    ) -> str:
        """Assemble generated code blocks into a Python module using LLM."""
        # Load structural and style knowledge
        pyspark_patterns = self.llm.load_knowledge("pyspark_patterns.md")
        skeleton_template = self.llm.load_knowledge("pyspark_skeleton_template.md")
        
        # Load package-specific analysis
        package_analysis = self._load_package_analysis(mapping.workflow_name)
        
        # Build assembly prompt
        prompt = self._build_module_assembly_prompt(
            mapping_name, code_blocks, mapping, pyspark_patterns, skeleton_template, package_analysis
        )
        
        # Generate the complete module via LLM
        response = self.llm.generate(prompt, self.system_prompt)
        
        # Extract code from response
        return self.llm.extract_code(response.text)
    
    def _build_module_assembly_prompt(
        self,
        mapping_name: str,
        code_blocks: list[str],
        mapping: Mapping,
        pyspark_patterns: Optional[str] = None,
        skeleton_template: Optional[str] = None,
        package_analysis: Optional[str] = None,
    ) -> str:
        """Build prompt for LLM to assemble the final PySpark module."""
        parts = []
        
        if pyspark_patterns:
            parts.append(f"## PySpark Best Practices\n{pyspark_patterns}")
        if skeleton_template:
            parts.append(f"## Standard PySpark skeleton template\n{skeleton_template}")
        
        if package_analysis:
            parts.append(f"## Package-Wide Business Logic & Domain Analysis\n{package_analysis}")
        
        block_text = chr(10).join([f"# Transformation Block {i+1}{chr(10)}{block}" for i, block in enumerate(code_blocks)])
        tx_types = ', '.join(sorted(set(tx.type for tx in mapping.transformations)))
        safe_filename = self._sanitize_filename(mapping_name)
        
        parts.append(f"""
## Task: Assemble Complete PySpark Module

You are assembling a complete, production-ready PySpark module for the SSIS package mapping: **{mapping_name}**

### Mapping Details:
- **Name:** {mapping_name}
- **Number of Transformations:** {len(mapping.transformations)}
- **Transformation Types:** {tx_types}

### Generated Code Blocks:
Below are the individual PySpark code blocks that have been generated for each transformation in this mapping. Your task is to combine and tailor them into a complete, executable Python module.

```python
{block_text}
```

## Requirements:

1. **Module Header:**
   - Add a comprehensive docstring describing the module.
   - Reference the original SSIS package mapping name.
   - Note that this was auto-generated.

2. **Imports:**
   - Import all necessary PySpark modules (SparkSession, DataFrame, functions as F, types, Window).
   - **CRITICAL**: Import the utils module: `from utils import *`.
   - Use the SSIS-equivalent utility functions from `utils.py` (like `iif`, `decode`, `nvl`, `to_date_ssis`) wherever possible to maintain logic fidelity.

3. **Transformation Functions:**
   - Organize the code blocks into well-named functions (snake_case).
   - Each function should accept one or more DataFrames and return one or more DataFrames (or a dictionary for multiple outputs).
   - **CRITICAL**: For transformations like `Conditional Split`, the function should return a dictionary of DataFrames corresponding to the split branches.
   - Include type hints and docstrings.

4. **Main Pipeline Function:**
   - Create a `run_mapping()` function that orchestrates the entire sequence.
   - It should chain the transformation functions in the correct order.
   - Include logging for start/end of the mapping.

5. **Main Entry Point:**
   - Add an `if __name__ == "__main__":` block to initialize Spark and call `run_mapping()`.

6. **Code Quality and Fidelity (CRITICAL):**
   - Follow PEP 8.
   - **STRICT FIDELITY**: Do NOT add logic or columns not present in the individual transformation blocks.
   - **NAMING**: Ensure the final output columns match the SSIS destination mapping EXACTLY.
   - Use broadcast joins for small lookups if information is available.
   - Add error handling and comments for complex logic.

## Output:
Generate the complete, executable PySpark module. The output should be ready to save as `{safe_filename}.py` and run in production.
""")
        
        return "\n\n".join(parts)
    
    # ─── Silver / Gold Tailoring via LLM ──────────────────────────
    
    def _generate_bronze_notebook(
        self,
        workflow: Workflow,
        output_dir: Path,
    ) -> Path:
        """Call LLM to generate a Bronze ingestion notebook from SSIS sources."""
        prompt = self._build_bronze_notebook_prompt(workflow)
        response = self.llm.generate(prompt, self.system_prompt)
        code = self.llm.extract_code(response.text)

        pkg_name = self._sanitize_filename(workflow.name)
        file_path = output_dir / f"bronze_{pkg_name}.py"
        file_path.write_text(code, encoding='utf-8')
        return file_path

    def _build_bronze_notebook_prompt(
        self,
        workflow: Workflow,
    ) -> str:
        """Build prompt for LLM to generate the Bronze ingestion notebook."""
        pkg_name = self._sanitize_filename(workflow.name)

        sources_text = ""
        if workflow.sources:
            src_lines = []
            for s in workflow.sources:
                src_lines.append(f"  - {s.name} (table/query: {s.sql_query or s.name})")
            sources_text = "\n### SSIS Source Components:\n" + chr(10).join(src_lines)

        conn_text = ""
        if workflow.connection_managers:
            conn_lines = []
            for cm in workflow.connection_managers:
                conn_lines.append(f"  - {cm.name} [{cm.creation_name}]")
            conn_text = "\n### Connection Managers (SSIS -> JDBC):\n" + chr(10).join(conn_lines)

        variables_text = ""
        if workflow.variables:
            var_lines = []
            for v in workflow.variables:
                var_lines.append(f"  - {v.namespace}::{v.name} = {v.value} [type: {v.data_type}]")
            variables_text = "\n### Package Variables:\n" + chr(10).join(var_lines)

        bronze_patterns = self.llm.load_knowledge("bronze_layer_patterns.md")
        connection_rules = self.llm.load_knowledge("connection_manager_rules.md")
        package_analysis = self._load_package_analysis(workflow.name)

        knowledge_section = ""
        if bronze_patterns:
            knowledge_section += f"\n## Bronze Layer Patterns Reference:\n{bronze_patterns}\n"
        if connection_rules:
            knowledge_section += f"\n## Connection Manager Rules Reference:\n{connection_rules}\n"
        if package_analysis:
            knowledge_section += package_analysis

        control_flow_text = self._build_control_flow_context(workflow)

        return f"""
## Task: Generate Bronze Layer PySpark Notebook

You are creating the **Bronze layer** notebook for the Medallion Architecture.

**Medallion Architecture:**
- **Bronze** = Raw source data extracted from operational databases via JDBC — THIS is what you're generating
- **Silver** = Cleansed and transformed data — generated in `silver_{pkg_name}.py`
- **Gold** = Aggregated, business-ready data — generated in `gold_{pkg_name}.py`

### Workflow: {workflow.name}
{sources_text}
{conn_text}
{variables_text}
{control_flow_text}
{knowledge_section}
## Requirements:

1. **Module Header:**
   - Docstring: "Bronze Layer - {workflow.name}"
   - Note: Medallion Architecture — Source Systems to Bronze
   - Auto-generation notice
   - SSIS equivalent: OLE DB Source components

2. **Imports:**
   - PySpark (SparkSession, DataFrame, functions as F, typing)
   - `concurrent.futures` for parallel extraction
   - Standard logging, typing

3. **Consolidated JDBC Configuration (CRITICAL):**
   - Implement ONE function `build_pipeline_jdbc_config() -> dict` that returns a flat dictionary containing all connection URLs (e.g., `source_url`, `target_url`, `npi_url`) and shared credentials.
   - **NO** separate `get_*_db_config` functions.
   - Credentials must NEVER be hardcoded. Use `os.environ.get`.

4. **EXTREME CONCISENESS (BLOCKER PREVENTION):**
   - **NO docstrings** for internal functions (`generic_extract`, `persist_to_bronze`).
   - Use a single `generic_extract` helper.
   - **DO NOT** create separate `extract_<table_name>` functions. Call `generic_extract` directly inside `run_bronze_pipeline` for every table.
   - This prevents file truncation and is MANDATORY.

5. **MANDATORY SIGNATURE**: `def run_bronze_pipeline(spark, jdbc_config, processing_date=None, batch_id=0):`
   - **Internal logic to split config**:
     ```python
     def run_bronze_pipeline(spark, jdbc_config, processing_date=None, batch_id=0):
         # Split flat jdbc_config into specific configs
         source_cfg = {{**jdbc_config, "url": jdbc_config["source_url"]}}
         target_cfg = {{**jdbc_config, "url": jdbc_config["target_url"]}}
         
         # Phase 1: Parallel extraction
         with ThreadPoolExecutor(max_workers=3) as executor:
             # submit persist_to_bronze(generic_extract(...), ...)
     ```

7. **Requirements Verification (MANDATORY):**
   - **NO ORDER BY:** Never include `ORDER BY` inside JDBC source queries. Spark's parallel reads ignore source ordering. Apply `.orderBy()` in PySpark if needed.
    - **NO SQL INJECTION:** Use variables or parameters for filtering. Never embed unsanitized user strings directly into SQL f-strings.
   - PEP 8, type hints, comprehensive docstrings.
   - **100% fidelity** to original SSIS extraction logic.

8. **Main Entry Point:**
   - `if __name__ == "__main__":` block
   - Initialize Spark with app name "{workflow.name} Bronze Layer"
   - Read JDBC credentials from environment variables
   - Call `run_bronze_pipeline(spark, jdbc_config)`
   - Stop Spark

9. **Code Quality:**
   - PEP 8, type hints, comprehensive docstrings
   - wrap extraction in try/except; re-raise on failure
   - Comments mapping each extraction back to its SSIS source component

## Output:
Generate the complete Bronze layer notebook as a single Python module.
"""

    def _tailor_silver_notebook(
        self,
        workflow: Workflow,
        silver_mappings: list,
        gold_mappings: list,
        silver_code_blocks: dict[str, str],
        output_dir: Path,
    ) -> Path:
        """Call LLM to tailor combined Silver mapping code into one cohesive Silver notebook."""
        prompt = self._build_silver_tailoring_prompt(workflow, silver_mappings, gold_mappings, silver_code_blocks)
        response = self.llm.generate(prompt, self.system_prompt)
        code = self.llm.extract_code(response.text)
        
        pkg_name = self._sanitize_filename(workflow.name)
        file_path = output_dir / f"silver_{pkg_name}.py"
        file_path.write_text(code, encoding='utf-8')
        return file_path
    
    def _tailor_gold_notebook(
        self,
        workflow: Workflow,
        gold_mappings: list,
        gold_code_blocks: dict[str, str],
        output_dir: Path,
        silver_mappings: list = None,
        silver_manifest: dict = None,
    ) -> Path:
        """Call LLM to tailor combined Gold mapping code into one cohesive Gold notebook.

        When gold_mappings is empty (no Aggregate transformations found), generates
        a minimal audit-only Gold module instead of letting the LLM hallucinate.

        Args:
            silver_manifest: dict from _extract_silver_manifest() containing
                'silver_tables' — the actual saveAsTable targets from the Silver notebook.
                When provided, used instead of deriving names from mapping names.
        """
        pkg_name = self._sanitize_filename(workflow.name)

        if not gold_mappings:
            # No aggregation mappings — generate a minimal audit-only Gold module
            logger.info("No Gold mappings found; generating audit-only Gold module")

            # Derive Silver table names: prefer silver_manifest (most accurate)
            # Fall back to sanitized mapping names if manifest not available
            silver_table_names: list[str] = []
            if silver_manifest and silver_manifest.get("silver_tables"):
                silver_table_names = silver_manifest["silver_tables"]
                logger.info("Gold: using silver_manifest table names", tables=silver_table_names)
            elif silver_mappings:
                for m in silver_mappings:
                    # Extract entity name from mapping: prefer the last segment after the
                    # last underscore-delimited SSIS task prefix, then lower-case.
                    # e.g. 'TSK_DF_OdsToStage' → 'address'  (from destination)
                    # Fallback: strip leading DFT_/TSK_/STG_ prefixes.
                    raw = self._sanitize_filename(m.name)  # e.g. 'tsk_df_odstostage'
                    # Prefer destination name if available
                    if m.destinations:
                        entity = self._sanitize_filename(m.destinations[0].name).lower()
                    else:
                        # Strip common SSIS task prefixes (dft_, tsk_, stg_, src_, lkp_)
                        import re as _re
                        entity = _re.sub(r'^(dft|tsk|stg|src|lkp|con|seq|rc|dc|cspt|mrg|scd|drv)_', '', raw, flags=_re.IGNORECASE).lower()
                    silver_table_names.append(entity)
                logger.warning(
                    "Gold: silver_manifest not available — falling back to destination/entity name heuristics.",
                    tables=silver_table_names,
                )
            tables_literal = repr(silver_table_names)
            audit_code = f'''"""Gold Layer - {workflow.name}

Medallion Architecture: Silver → Gold
Auto-generated — No aggregate transformations found in SSIS package.
This module provides audit-only pass-through from Silver to Gold.
"""
import logging
from datetime import datetime
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)


def run_gold_pipeline(spark: SparkSession) -> dict:
    """Pass-through Gold pipeline — no aggregation transformations in source SSIS package.

    This is an audit-only module. Add explicit table names to the list below.
    """
    results = {{}}
    # Silver table names auto-derived from the package's Silver mappings
    tables_to_promote = {tables_literal}

    logger.info(f"Gold pass-through: {{len(tables_to_promote)}} tables")

    for table_name in tables_to_promote:
        try:
            # Table name must be the sanitized entity name (e.g., 'address'), not the SSIS task name
            df = spark.table(f"silver.{{table_name.lower()}}")
            df = df.withColumn("_gold_load_timestamp", F.current_timestamp())
            df = df.withColumn("_gold_source", F.lit(f"silver.{{table_name}}"))
            df.write.mode("overwrite").saveAsTable(f"gold.{{table_name}}")
            row_count = df.count()
            logger.info(f"Gold pass-through: {{table_name}} — {{row_count}} rows")
            results[table_name] = df
        except Exception as e:
            logger.error(f"Gold pass-through failed for {{table_name}}: {{e}}", exc_info=True)
            raise

    return results


if __name__ == "__main__":
    spark = SparkSession.builder.appName("{workflow.name} Gold Layer").getOrCreate()
    try:
        run_gold_pipeline(spark)
    finally:
        spark.stop()
'''
            gold_path = output_dir / f"gold_{pkg_name}.py"
            gold_path.write_text(audit_code, encoding="utf-8")
            return gold_path

        prompt = self._build_gold_tailoring_prompt(
            workflow, gold_mappings, gold_code_blocks,
            silver_mappings=silver_mappings or [],
            silver_manifest=silver_manifest,
        )
        response = self.llm.generate(prompt, self.system_prompt)
        code = self.llm.extract_code(response.text)
        file_path = output_dir / f"gold_{pkg_name}.py"
        file_path.write_text(code, encoding='utf-8')
        return file_path


    def _build_silver_tailoring_prompt(
        self,
        workflow: Workflow,
        silver_mappings: list,
        gold_mappings: list,
        silver_code_blocks: dict[str, str],
    ) -> str:
        """Build prompt for LLM to tailor Silver notebook."""
        pkg_name = self._sanitize_filename(workflow.name)
        mapping_details = []
        for m in silver_mappings:
            tx_types = ', '.join(sorted(set(tx.type for tx in m.transformations)))
            mapping_details.append(
                f"### Mapping: {m.name}\n"
                f"- Description: {m.description or 'N/A'}\n"
                f"- Transformations: {len(m.transformations)}\n"
                f"- Types: {tx_types}"
            )
        
        code_sections = []
        for name, code in silver_code_blocks.items():
            code_sections.append(f"# === Mapping: {name} ===\n{code}")
        
        combined_code = "\n\n".join(code_sections)
        mapping_detail_text = chr(10).join(mapping_details)
        
        sources_text = ', '.join([s.name for s in workflow.sources]) if workflow.sources else 'N/A'
        
        # Load knowledge docs for Silver context (focused set — avoid prompt bloat)
        variable_rules = self.llm.load_knowledge("variable_rules.md")
        connection_manager_rules = self.llm.load_knowledge("connection_manager_rules.md")
        silver_patterns = self.llm.load_knowledge("silver_layer_patterns.md")
        # Omit env_references, project_params, fabric_patterns here — already in system prompt
        transformation_rules = self.llm.load_knowledge("../transformation_rules.md")
        
        # Build variables context
        variables_text = ""
        if workflow.variables:
            var_lines = []
            for v in workflow.variables:
                flags = []
                if v.readonly:
                    flags.append("readonly")
                if v.is_expression:
                    flags.append("expression")
                flag_str = f" ({', '.join(flags)})" if flags else ""
                var_lines.append(f"  - {v.namespace}::{v.name} = {v.value} [type: {v.data_type}]{flag_str}")
            variables_text = "\n### Package Variables:\n" + chr(10).join(var_lines)
        
        # Build connection managers context
        conn_text = ""
        if workflow.connection_managers:
            conn_lines = []
            for cm in workflow.connection_managers:
                conn_lines.append(f"  - {cm.name} [{cm.creation_name}]")
            conn_text = "\n### Connection Managers:\n" + chr(10).join(conn_lines)
        
        knowledge_section = ""
        if variable_rules:
            knowledge_section += f"\n## Variable Conversion Rules Reference:\n{variable_rules}\n"
        if connection_manager_rules:
            knowledge_section += f"\n## Connection Manager Rules Reference:\n{connection_manager_rules}\n"
        if silver_patterns:
            knowledge_section += f"\n## Silver Layer Patterns Reference:\n{silver_patterns}\n"
        if transformation_rules:
            knowledge_section += f"\n## Transformation Rules:\n{transformation_rules}\n"

        package_analysis = self._load_package_analysis(workflow.name)
        if package_analysis:
            knowledge_section += package_analysis

        control_flow_text = self._build_control_flow_context(workflow)

        return f"""
## Task: Tailor Silver Layer PySpark Notebook

You are creating the **Silver layer** notebook for the Medallion Architecture.

**Medallion Architecture:**
- **Bronze** = Raw source data extracted from operational databases — generated in `bronze_{pkg_name}.py`, tables stored as `bronze.<table>`. The Silver layer reads from these tables.
- **Silver** = Cleansed and transformed data — THIS is what you're generating
- **Gold** = Aggregated, business-ready data — generated separately

### Workflow: {workflow.name}
### Bronze Sources (read from bronze.* tables): {sources_text}
### Targets: Silver tables (transformed output)
{variables_text}
{conn_text}
{control_flow_text}

### Silver Mappings:
{mapping_detail_text}

### Generated Code (per-mapping, to be combined):
Below are the individually generated PySpark code blocks for each Silver mapping.
Your task is to combine and tailor them into ONE cohesive Silver layer notebook.

```python
{combined_code}
```
{knowledge_section}
## Requirements:

1. **MANDATORY Orchestration (ABSOLUTE TOP PRIORITY):**
   - **[CRITICAL]** implement `run_silver_pipeline(spark, processing_date, jdbc_config) -> dict` at the VERY TOP of the file.
    - **RADICAL CONCISENESS (TRUNCATION PREVENTION)**:
      - **FORBIDDEN**: NO docstrings for ANY function.
      - **FORBIDDEN**: NO mapping comments or redundant logic explanations.
      - **TABLE NAMING (CRITICAL)**: Always read from actual entity tables (e.g., `bronze.member`, `bronze.claims`). 
      - **NO COMPONENT NAMES**: NEVER use SSIS component/task names like `bronze.SRC_Member` or `bronze.OLE_DB_Source` for table names.
      - **FORBIDDEN**: NO separate `read_bronze_*` functions. Use a single line `spark.table("bronze.<table>").filter(...)` inside the orchestrator.
      - **ARCHITECTURAL BOUNDARY (CRITICAL)**: This is the **SILVER** layer. Do NOT implement any aggregations or summary-level logic (tasks marked `[GOLD]` in the hierarchy). Focus ONLY on cleansing and transformation.
      - **LOOKUP FIDELITY (CRITICAL)**: Only include columns explicitly selected in the SSIS Lookup Output metadata. Do NOT propagate "extra" columns (e.g., `taxonomy_code`) from reference tables just because they are available.
   - This is necessary to ensure the file is 100% generated without truncation.

4. **Transformation Functions:**
   - **Function signature**: `def transform_*(input_df: DataFrame) -> DataFrame`.
   - **[MANDATORY] NO docstrings.** NO mapping comments.
   - Implement correct source reading (e.g., `spark.table('silver.member_claims')`).
   - **NO `AttributeError` PySpark functions**: No `F.right()`, `F.left()`, `F.replace()`. Use `F.substring()`, `F.regexp_replace()`, `F.lpad()`.
     ```python
     # SSIS: RIGHT("00000000000" + REPLACE(REPLACE(TRIM(code), "-", ""), " ", ""), 11)
     # Standardize code to 11 digits by stripping separators then left-padding
     df = df.withColumn(
         "code_standardized",
         F.lpad(F.regexp_replace(F.regexp_replace(F.trim(F.col("source_code")), r"-", ""), r" ", ""), 11, "0")
     )
     ```
   - Organize ALL transformation code into well-named functions
   - Preserve ALL SSIS transformation logic exactly (100% accuracy)
   - Keep column names, expressions, data types faithful to SSIS
   - **Lookup transforms**: Include ALL return columns from the SSIS output column list (e.g. if lookup returns related_name, related_type, category_code — ALL three MUST appear)
   - **Lookup no-match**: When SSIS redirects unmatched rows, use inner + left_anti split, add null columns for lookup outputs to unmatched, apply downstream transforms to the correct path, then schema-align and union. NEVER use an empty placeholder DataFrame.
    - **AGGREGATION FIDELITY (CRITICAL)**: ONLY group by columns explicitly marked as GroupBy inputs in the SSIS metadata. Do NOT add logical dimensions (year, month, batch_id) unless explicitly present in the component.
   - **CRITICAL — Forbidden PySpark functions**: `F.right()`, `F.left()`, and `F.replace()` do NOT exist in PySpark and will cause `AttributeError` at runtime. Use the correct equivalents:
     - String replacement → `F.regexp_replace(col, pattern, replacement)`
     - Right N characters → `F.substring(col, -n, n)` or `F.expr("right(col, n)")`
     - Left N characters → `F.substring(col, 1, n)`
     - Left-pad → `F.lpad(col, length, pad)`
    - **CRITICAL — COMPLETENESS RULE**: You MUST implement **ALL** Data Flow Tasks listed in the analysis document. Do NOT skip any mappings. If a mapping is listed in the header but missing implementation, the pipeline is BROKEN.
    - **THREAD SAFETY / AUDIT RULE**: Do NOT use global module-level variables for row counts (e.g., `row_count_members = 0`). This is a race condition risk when using `ThreadPoolExecutor`. Instead, ensure each `run_dft_*` function returns its row count (or the DataFrame itself), and have `run_silver_pipeline` collect and return them in a results dictionary.
    - **NO HARDCODED DATES**: Never hardcode `User::ProcessingDate` as a module-level constant (e.g., `processing_date = datetime(2026,1,11)`). Always pass it as a parameter `processing_date` through the function chain starting from `main.py`.
      - This function MUST orchestrate the execution of all `run_dft_*` functions.
      - It MUST respect the `task_filter` (e.g., if `task_filter='members'`, only run the members DFT).
      - It MUST return a dictionary of row counts: `{{ "members": count1, "claims": count2, ... }}`.
    - **CRITICAL — Special character stripping**:
      Before padding or formatting any code-like field (e.g., product codes, keys), strip ALL non-alphanumeric characters
      using `F.regexp_replace()` with regex `r'[-\\s*]'`.
      ```python
      # Example Transformation Logic:
      cleaned = F.regexp_replace(F.trim(F.col('input_code')), r'[-\\s*]', '')
      result  = F.lpad(cleaned, 10, '0')
      ```
     Apply this pattern to ANY field that requires standardisation (product codes, transaction codes,
     etc.) — do NOT assume input data is clean.

4b. **CRITICAL — Lookup Transformations for Batch Metadata:**
    - If the SSIS package performs a lookup against `[process].[Batch]` or similar to get a `BatchId`:
      - Implement a `lookup_batch(input_df, spark, jdbc_config)` function.
      - Read the reference table via JDBC using the passed-in `jdbc_config`.
      - Return all input columns + `BatchId` and other metadata.
    - If no lookup is present but `BatchId` is required, obtain it from the Bronze layer metadata or generate it.

4c. **CRITICAL — Lookup Functions that read reference tables via JDBC:**
   - Any lookup function that reads a reference table from a database MUST accept `jdbc_config: dict` as an explicit parameter. NEVER reference a bare `jdbc_url` variable that is not in scope.
   - **WRONG**: `def lookup_ref(input_df, spark): ... spark.read.format("jdbc").option("url", jdbc_url)` — `jdbc_url` is undefined → `NameError`
   - **CORRECT**: `def lookup_ref(input_df: DataFrame, spark: SparkSession, jdbc_config: dict) -> DataFrame: ... .option("url", jdbc_config["url"])...`
   - The `run_silver_pipeline()` function must pass `jdbc_config` through to every lookup function that needs it.
   - SQL strings inside `.option("query", ...)` or `.option("dbtable", "(SELECT ...) alias")` MUST be ANSI SQL — NOT T-SQL:
     - String concat: use `CONCAT(a, b)` not `a + b`
     - Null check: use `COALESCE(col, val)` not `ISNULL(col, val)`
     - No `TOP N`, use PySpark `.limit(N)` instead
     - No `GETDATE()`, use `CURRENT_TIMESTAMP`
     - No `CONVERT(type, col)`, use `CAST(col AS type)`
   - **JDBC SUBQUERY WRAPPING**: Always wrap JDBC subqueries in parentheses: `.option("dbtable", "(SELECT * FROM schema.table) as refTable")`.

    - **SELECTION**: Preserve all source columns (`src.*`) and include the primary/surrogate key from the target (aliased) to identify matches/duplicates in downstream steps.

4e. **Conditional Split Transformations:**
    - If the SSIS package uses a Conditional Split (e.g., `SPL_FilterValid`):
      - Implement a transformation function for EACH output branch (e.g., `split_valid_rows`, `split_invalid_rows`).
      - Use `.filter()` for the main condition.
      - Use `.filter(~condition)` or `.left_anti()` for the default/error branch.
      - Ensure BOTH branches are used if they lead to different destinations or subsequent steps.

4f. **Audit Variable Updates (INCREMENTAL):**
    - The SSIS package often updates variables like `EXTRACT_COUNT` or `LOAD_COUNT`.
    - In PySpark, perform a `.count()` after major steps (e.g., extraction, transformation, load).
    - Increment the global capitalized variables (e.g., `EXTRACT_COUNT += df.count()`) AND log the counts immediately.

4e. **Conditional Split Transformations (MANDATORY SPLITTING):**
    - Conditional Split functions MUST actually filter rows into separate DataFrames based on the SSIS conditions (e.g., `src_id.isNull()` for "New Records").
    - **RETURN DICTIONARY**: Return a dictionary of DataFrames where keys match SSIS output branch names.
    - `run_silver_pipeline()` must process these branches according to the control flow (e.g., only inserting "New" records).

4g. **MANDATORY Transformation Fixes (BLOCKER PREVENTION):**
    - **[MANDATORY]** Implement `transform_ndc_standardization(input_df: DataFrame) -> DataFrame` logic:
        - `cleaned = F.regexp_replace(F.trim(F.col('ndc_code')), r'[-\\s*]', '')`
        - `ndc_11 = F.lpad(cleaned, 11, '0')` (Standardised)
        - `ndc_formatted = concat 5-4-2 hyphenated`
        - `rx_year = F.year(F.col('fill_date'))`
    - **[MANDATORY]** Implement the FULL `derive_claim_type` (or `transform_der_claim_type`) logic.
    - **EXTREME CONCISENESS**: Do NOT write long docstrings. Do NOT write redundant mapping comments. Use short variable names. Every token saved prevents file truncation.
    - NO docstrings for internal helper functions.

5. **CRITICAL — Derived Column transforms MUST use `withColumn()`, NOT `select()`:**
   - An SSIS Derived Column transformation **adds** new columns to the existing row. It never drops existing columns.
   - Inside every `transform_*` function, use `df.withColumn("new_col", expr)` to ADD the derived column while keeping ALL input columns.
   - **WRONG** (drops all other source columns — NEVER do this inside a transform function):
     ```python
     # ❌ BAD — drops all 17 source columns, only returns 3
     return input_df.select(
         F.col("entity_id"),
         F.when(...).alias("status_standardized"),
         F.when(...).alias("code_formatted")
     )
     ```
   - **CORRECT** (adds derived columns alongside all existing columns):
     ```python
     # ✅ GOOD — all 17 source columns + 2 new derived columns = 19 columns out
     df = input_df.withColumn("status_standardized", F.when(...).otherwise("Other"))
     df = df.withColumn("code_formatted", F.when(...).otherwise(...))
     return df  # returns ALL original columns plus the new derived ones
     ```
   - Both the original column (e.g. `status`) AND the derived column (e.g. `status_standardized`) must be present in the output.
   - Downstream transformations may need the original columns for joins, lookups, or further expressions.

5b. **Destination Column Projection (write step only):**
   - ONLY at the final `saveAsTable()` write step should you narrow the DataFrame to destination columns.
   - Apply `.select([only destination cols])` immediately before `write.mode("overwrite").saveAsTable(...)`, not inside the transform function.
   - If the SSIS destination maps 17 columns, the write step selects exactly those 17 columns (which may include both original and derived columns).
   - Pattern:
     ```python
     silver_df = transform_entity(bronze_df)       # returns all cols incl. derived
     write_df = silver_df.select(                  # narrow only at write time
         "entity_id", "code", "name", "status",
         "status_standardized", "code_formatted", ...
     )
     write_df.write.mode("overwrite").saveAsTable("silver.entity")
     ```

6. **Silver Pipeline Orchestration (MANDATORY COMPLETENESS):**
   - Create `run_silver_pipeline(spark: SparkSession, processing_date: str = None, jdbc_config: dict = None) -> dict[str, DataFrame]`
   - **ORCHESTRATION COMPLETENESS RULE**: Every `transform_*`, `lookup_*`, `derived_*`, and `split_*` function defined in the module MUST be called exactly once in `run_silver_pipeline()`.
   - Verify function counts: ensure (count of `def transform_*`) == (count of `transform_*()` calls in orchestration).
   - **JDBC PASSTHROUGH**: If any transform/lookup function accepts `jdbc_config`, then `run_silver_pipeline()` MUST also accept it and forward it correctly.
   - Orchestrate all transformation functions in order, following the SSIS precedence constraints.
   - Handle parallel execution for independent Data Flow Tasks using `ThreadPoolExecutor`:
     ```python
     # Example parallel execution
     from concurrent.futures import ThreadPoolExecutor, as_completed
     with ThreadPoolExecutor(max_workers=2) as executor:
          future_a = executor.submit(transform_a, spark)
          future_b = executor.submit(transform_b, spark)
          for future in as_completed([future_a, future_b]):
              future.result()  # re-raises on failure
     ```
   - Add logging (`logger.info`) for each mapping step with row count
   - Return dict of silver_table_name to DataFrame

6b. **Performance — Caching and Broadcasts:**
   - **Cache lookup reference tables** that are loaded once and reused across multiple transforms:
     ```python
     # Load and cache reference table for use in multiple lookup transforms
     ref_table_df = spark.table("bronze.reference_table").cache()
     ref_table_df.count()  # trigger materialisation

     try:
         result1 = lookup_reference(df_a, spark, ref_table_df)
         result2 = lookup_reference(df_b, spark, ref_table_df)
     finally:
         ref_table_df.unpersist()  # release memory when done
     ```
   - **Broadcast small reference tables** (< 10 MB) in all lookup joins:
     ```python
     result_df = input_df.join(
         F.broadcast(reference_df),   # avoids shuffle for small tables
         on="join_key", how="left"
     )
     ```
   - Do NOT cache large fact tables — let Spark manage partitioning.

7. **Row Count Logging (MANDATORY):**
   - Every mapping function MUST log row counts using `df.count()` after the final transform
   - This replaces SSIS Row Count (RC_) components
   - Pattern: `logger.info(f"DFT_TransformEntity completed: {{row_count}} rows processed")`

 8. **Silver Output (JDBC DESTINATIONS):**
    - **CRITICAL**: If the SSIS destination is a SQL Server table (e.g. `[dbo].[Address]`), use `.write.jdbc()` to save data back to that table.
    - Use `append` mode for incremental loads after conditional split.
    - **ONLY** at final write step select the subset of columns mapped in SSIS.
    - Add audit columns: `_load_timestamp`, `_source_system`.
    - If no JDBC sink is required, fall back to `df.write.mode("append").saveAsTable(f"silver.table_name")`.

9. **Main Entry Point:**
   - `if __name__ == "__main__":` block
   - Initialize Spark, run silver pipeline, write tables, stop Spark

10. **Error Handling (SSIS Error Output equivalent):**
   - Wrap the body of EVERY `transform_*`, `lookup_*`, and `split_*` function in a `try/except`:
     ```python
     def transform_entity(spark: SparkSession) -> DataFrame:
         try:
             # Identify correct Bronze table from persist_to_bronze() in bronze_*.py
             df = spark.table("bronze.entity") 
             ...

             df = df.withColumn("status_standardized", ...)
             row_count = df.count()
             logger.info(f"DFT_TransformEntity: {{row_count}} rows")
             return df
         except Exception as e:
             logger.error(f"DFT_TransformEntity failed: {{e}}", exc_info=True)
             raise  # re-raise so run_silver_pipeline can catch and route
     ```
   - **Dead-letter / error rows**: For row-level data quality issues (bad casts, nulls in NOT NULL
     columns), use a tolerant schema then separate good from bad rows:
     ```python
     good_df = df.filter(F.col("entity_id").isNotNull())
     bad_df  = df.filter(F.col("entity_id").isNull()) \
                 .withColumn("_error_reason", F.lit("entity_id is null")) \
                 .withColumn("_error_timestamp", F.current_timestamp())
     bad_df.write.mode("append").saveAsTable("error.silver_entity_rejects")
     ```
   - PEP 8, type hints, comprehensive docstrings
   - Comments explaining business logic from SSIS
   - 100% faithful to original SSIS transformation logic

11. **Data Quality Validation (SSIS implicit validation equivalent):**
   - For EVERY Silver mapping, generate a `validate_<entity>_data(df: DataFrame, spark: SparkSession) -> DataFrame` function called BEFORE writing to the Silver table.
   - The validation function MUST cover:
     a. **NULL checks** on required / primary-key columns (infer from SSIS source schema or column names ending in `_id`, `_code`)
     b. **Duplicate detection** on the primary key — log a warning + write duplicates to error table
     c. Return ONLY valid rows (filter out NULLs in required fields)
   - Pattern (adapt column names to each entity):
     ```python
     def validate_entity_data(df: DataFrame, spark: SparkSession) -> DataFrame:
         \"\"\"Data quality validation for Silver entity table.
         Equivalent to SSIS NULL handling and type validation in transformations.\"\"\"
         total = df.count()

         # 1. NULL check on required fields
         null_df = df.filter(
             F.col("entity_id").isNull() |
             F.col("update_date").isNull()
         ).withColumn("_error_reason", F.lit("Required field is NULL")) \
          .withColumn("_error_timestamp", F.current_timestamp())

         null_count = null_df.count()
         if null_count > 0:
             logger.warning(f"validate_entity_data: {{null_count}}/{{total}} rows have NULL required fields")
             null_df.write.mode("append").saveAsTable("error.silver_entity_rejects")

         # 2. Duplicate detection on primary key
         dup_df = df.groupBy("entity_id").count().filter(F.col("count") > 1)
         dup_count = dup_df.count()
         if dup_count > 0:
             logger.warning(f"validate_entity_data: {{dup_count}} duplicate entity_id values detected")
             dup_keys = dup_df.select("entity_id")
             df.join(dup_keys, on="entity_id", how="inner") \
               .withColumn("_error_reason", F.lit("Duplicate primary key")) \
               .withColumn("_error_timestamp", F.current_timestamp()) \
               .write.mode("append").saveAsTable("error.silver_entity_rejects")

         # 3. Return valid rows only (non-NULL required fields; keep duplicates with warning)
         valid_df = df.filter(F.col("entity_id").isNotNull())
         valid_count = valid_df.count()
         logger.info(f"validate_entity_data: {{valid_count}} valid rows returned ({{total - valid_count}} rejected)")
         return valid_df
     ```
   - Call `validate_<entity>_data(df, spark)` after all transforms and BEFORE `df.write...saveAsTable()`.
   - Use `error.silver_<table>_rejects` as the error table naming convention.
   - Adapt the required-field list and primary key to each entity's actual columns.

## MANDATORY FUNCTION CHECKLIST — every Silver module MUST include all of the following:

For EACH ConditionalSplit component found in the SSIS data flow:
  ☑ `split_<entity>_rows(df: DataFrame) -> Dict[str, DataFrame]`
    - Must return a dict with a key per SSIS output branch (e.g. `"condition_error"`, `"default"`).
    - The `"condition_error"` branch uses the exact SSIS condition expression.
    - The `"default"` branch is `~condition` (all remaining rows).

For audit column injection (DerivedColumn components adding BatchId, BatchRunId, SourceSystem, etc.):
  ☑ `transform_audit_columns(df: DataFrame, batch_id=None, batch_run_id=None) -> DataFrame`
    - Must add: BatchId, BatchRunId, SourceSystem, _LoadStatus ("READY FOR EDR"), _OperationFlag ("I"), _BatchStep (1), SourceTB (entity abbreviation), _silver_load_timestamp.
    - Applied to BOTH the default branch AND the error branch.

For error output DerivedColumn (DFT_DC_AddColumnsForErrorLog or similar):
  ☑ `derive_error_log_columns(df: DataFrame, entity_name: str) -> DataFrame`
    - Must add: ErrorDescription=NULL, ObjectName=entity_name, ReferenceObjectName=NULL, ReferenceIdentifier=NULL, Identifier=<PrimaryKeyColumn>.
    - Applied ONLY to the error branch, BEFORE resolve_error_descriptions.

For Script Components (C# or VB) that resolve error codes:
  ☑ `resolve_error_descriptions(df: DataFrame) -> DataFrame`
    - Define `_SSIS_ERROR_CODES: dict[int, str]` at module level with the standard SSIS error codes.
    - Use `F.create_map(...)` built from `_SSIS_ERROR_CODES` for vectorised lookup.
    - Replace NULL ErrorDescription using the map, preserve non-null values.
    - Applied ONLY to the error branch, AFTER derive_error_log_columns.

For row count tracking (RowCount SSIS components):
  ☑ Module-level variables: `EXTRACT_COUNT = 0`, `ERROR_COUNT = 0`, `LOAD_COUNT = 0`.
  ☑ In `run_silver_pipeline`: set EXTRACT_COUNT = joined_df.cache().count() AFTER the merge join.
  ☑ Set ERROR_COUNT = error_df.cache().count() AFTER conditional split.
  ☑ Derive LOAD_COUNT = EXTRACT_COUNT - ERROR_COUNT.
  ☑ `run_silver_pipeline` MUST return `{{"extract_count": EXTRACT_COUNT, "error_count": ERROR_COUNT, "load_count": LOAD_COUNT}}`.

For writing Silver data (OLE DB Destination → Silver table):
  ☑ `write_to_silver_<entity>(df: DataFrame) -> int`
    - mode = "overwrite" + overwriteSchema=true (per-batch full replace).
    - Returns the row count written.
    - Writes the UNION of "default" + "new" paths (all non-error rows).

For writing error rows (OLE DB Destination → BatchErrors table):
  ☑ `write_to_batch_errors(df: DataFrame, entity: str) -> int`
    - mode = "append" (accumulates errors across batches).
    - Table name: `error.silver_<entity>_rejects`.
    - Only write if row_count > 0.
    - Returns the error row count.

For the orchestrator:
  ☑ `run_silver_pipeline(spark: SparkSession, batch_identifier=None, batch_id=None, batch_run_id=None) -> dict`
    - Signature MUST include batch_identifier, batch_id, batch_run_id parameters.
    - Orchestration order: read → derive_batch_columns → lookup_batch → merge_join → [EXTRACT_COUNT] → split → [ERROR_COUNT] → transform_audit (both branches) → derive_error_log (error branch) → resolve_error_descriptions (error branch) → [LOAD_COUNT = EXTRACT_COUNT - ERROR_COUNT] → write_to_silver → write_to_batch_errors → return metrics.
    - Returns dict: `{{"extract_count": ..., "error_count": ..., "load_count": ...}}`.

## Output:
Generate the complete Silver layer notebook as a single Python module.
"""
    
    def _build_gold_tailoring_prompt(
        self,
        workflow: Workflow,
        gold_mappings: list,
        gold_code_blocks: dict[str, str],
        silver_mappings: list = None,
        silver_manifest: dict = None,
    ) -> str:
        """Build prompt for LLM to tailor Gold notebook."""
        pkg_name = self._sanitize_filename(workflow.name)
        mapping_details = []
        for m in gold_mappings:
            tx_types = ', '.join(sorted(set(tx.type for tx in m.transformations)))
            mapping_details.append(
                f"### Mapping: {m.name}\n"
                f"- Description: {m.description or 'N/A'}\n"
                f"- Transformations: {len(m.transformations)}\n"
                f"- Types: {tx_types}"
            )

        code_sections = []
        for name, code in gold_code_blocks.items():
            code_sections.append(f"# === Mapping: {name} ===\n{code}")

        combined_code = "\n\n".join(code_sections)
        mapping_detail_text = chr(10).join(mapping_details)

        targets_text = ', '.join([t.name for t in workflow.targets]) if workflow.targets else 'N/A'

        control_flow_text = self._build_control_flow_context(workflow)

        # Load Gold layer knowledge
        gold_patterns = self.llm.load_knowledge("gold_layer_patterns.md")
        fabric_patterns = self.llm.load_knowledge("microsoft_fabric_patterns.md")

        gold_knowledge_section = ""
        if gold_patterns:
            gold_knowledge_section += f"\n## Gold Layer Patterns Reference:\n{gold_patterns}\n"
        if fabric_patterns:
            gold_knowledge_section += f"\n## Microsoft Fabric Target Architecture:\n{fabric_patterns}\n"

        # Build Silver Table Manifest text using actual manifest if available
        if silver_manifest and "silver_table_fqns" in silver_manifest:
            silver_manifest_text = (
                "\n### Silver Table Manifest (Actual tables written by Silver):\n"
                "The Gold layer MUST read from these tables (use exactly these names):\n"
                + "\n".join([f"  - {t}" for t in silver_manifest["silver_table_fqns"]])
            )
        else:
            silver_table_manifest = []
            if silver_mappings:
                for m in silver_mappings:
                    safe = self._sanitize_filename(m.name).lower()
                    silver_table_manifest.append(
                        f"  - silver.<table_derived_from_{safe}> (Silver mapping: {m.name})"
                    )
            silver_manifest_text = (
                "\n### Silver Table Manifest (Estimated):\n"
                + "\n".join(silver_table_manifest)
                if silver_table_manifest
                else "### Silver Table Manifest: (no silver mappings found)"
            )

        knowledge_section = ""
        if gold_patterns:
            knowledge_section = f"\n## Gold Layer Patterns Reference:\n{gold_patterns}\n"
        
        package_analysis = self._load_package_analysis(workflow.name)
        if package_analysis:
            knowledge_section += package_analysis

        return f"""
## Task: Tailor Gold Layer PySpark Notebook

You are creating the **Gold layer** notebook for the Medallion Architecture.

**Medallion Architecture:**
- **Bronze** = Raw source data extracted from operational databases — generated in `bronze_{{{{pkg_name}}}}.py`
- **Silver** = Cleansed and transformed data — generated in `silver_{{{{pkg_name}}}}.py`
- **Gold** = Aggregated, business-ready data — THIS is what you're generating

### Workflow: {workflow.name}
{control_flow_text}
{silver_manifest_text}
### Outputs: Gold tables → {targets_text}

### Gold Mappings:
{mapping_detail_text}

{knowledge_section}

### Generated Code (per-mapping, to be combined):
Below are the individually generated PySpark code blocks for each Gold mapping.
Your task is to combine and tailor them into ONE cohesive Gold layer notebook.

```python
{combined_code}
```

## Requirements:

1. **MANDATORY Orchestration (ABSOLUTE TOP PRIORITY):**
   - **[CRITICAL]** implement `run_gold_pipeline(spark) -> dict` at the VERY TOP of the file.
   - This function MUST call all transformation/aggregation functions defined in the "Generated Code" section below.
   - Use `ThreadPoolExecutor` to run independent aggregations in parallel.
   - Sequential calls should only be used where a dependency (precedence constraint) exists.
   - For EACH mapping, ensure `.write.mode("overwrite").saveAsTable("gold.<table_name>")` is called.
   - Return a metrics dict (e.g. `{{ "mapping_name": count, ... }}`).
   - **RADICAL CONCISENESS**:
     - **FORBIDDEN**: NO docstrings.
     - **FORBIDDEN**: NO redundant mapping comments.
     - **FORBIDDEN**: NO separate source readers. Use `spark.table()` directly in orchestrators.

1a. **CRITICAL — Aggregation Output Column Names (ABSOLUTE FIDELITY):**
   - Use the **EXACT output column names** from the SSIS Aggregate component metadata.
   - The metadata JSON in each transformation's Fields section includes OUTPUT fields with their exact `name`, `agg_type`, and `src_col`.
   - **Do NOT rename columns**. Use `.alias("<exact_name_from_metadata>")` matching the metadata `name` field.
   - Example: if metadata says `name: "inpatient_claims", agg_type: "SUM", src_col: "inpatient_count"`,
     generate `F.sum("inpatient_count").alias("inpatient_claims")` — NOT `total_inpatient_claims`.
   - Example: if metadata says `name: "first_encounter_date", agg_type: "MIN", src_col: "service_date_start"`,
     generate `F.min("service_date_start").alias("first_encounter_date")` — NOT `min_service_date`.
   - Include ALL aggregation output columns from the metadata — do not omit any.

1b. **CRITICAL — Memory Management (N-03):**
   - After calling `.cache()` on a DataFrame and writing it with `.saveAsTable()`, you MUST call `.unpersist()` immediately afterwards.
   - Pattern (MANDATORY):
     ```python
     result_df = compute_aggregation(df).cache()
     result_df.write.mode("overwrite").saveAsTable("gold.table_name")
     result_df.unpersist()  # ← MANDATORY — release cached memory
     ```
   - NEVER return a cached DataFrame without unpersisting after the write.

2. **Imports:**
   - Import PySpark modules (SparkSession, DataFrame, functions as F, types, Window)
   - **CRITICAL**: `from utils import *`
   - `from concurrent.futures import ThreadPoolExecutor, as_completed`
   - Standard logging, typing imports

3. **Source Reading Functions (CRITICAL — Silver Table Names):**
   - Read from Silver layer tables using `spark.table("silver.<table_name>")`
    - **TABLE NAMING (CRITICAL)**: Always promote actual entity tables (e.g., `bronze.address` -> `silver.address` -> `gold.address`). 
    - **NO TASK NAMES**: Do NOT use SSIS task names like `silver.DFT_GenerateBatches` or `silver.tsk_df_odstostage` for table names. Never derive `tables_to_promote` from SSIS container or data-flow task names.
    - **Silver Table Manifest**: Reference the exact Silver table names from the manifest provided above. If a mapping is for "Address", use `silver.address`.
      For example if Silver writes `silver.address`, the Gold reader
      must use `spark.table("silver.address")` — NOT `spark.table("silver.dft_generatebatches")`.
    - **`tables_to_promote` — MANDATORY format**:
      ```python
      # ❌ WRONG — SSIS task/container names, not real table names
      tables_to_promote = ['dft_generatebatches', 'tsk_df_odstostage']

      # ✅ CORRECT — actual entity names matching Silver saveAsTable() calls
      tables_to_promote = ['batch', 'address']  # from silver_manifest silver_tables list
      ```
    - Cross-reference the Silver manifest `silver_tables` list provided above.
      If no manifest is available, use the entity/destination name (e.g. 'address', 'batch'),
      never the Data Flow Task container name.
   - Check the Silver module `saveAsTable()` calls (lines containing `silver.`) for exact names.
   - Name functions: `read_silver_<table>(spark)`
   - **UNION ALL Sources**: When the SSIS source SQL contains UNION ALL across tables (e.g.,
     table_a + table_b), build the combined DataFrame in PySpark using `.unionByName()`. Do NOT
     read from a non-existent pre-combined table. Preserve ALL columns including literal type
     tags (e.g., `F.lit('tag_value').alias('tag_column')`).
   - **JOINED Sources**: When a Gold mapping requires data from multiple
      Silver tables (e.g. entity + transactions joined together), DO NOT invent a single
      non-existent Silver table. Instead, read each Silver table
      separately and JOIN them in the Gold function:
      ```python
      # ❌ WRONG — silver.combined_view doesn't exist
      df = spark.table("silver.combined_view")

      # ✅ CORRECT — join individual Silver tables that DO exist
      entity_df      = spark.table("silver.entity")
      transaction_df = spark.table("silver.transactions")
      combined_df = entity_df \
          .join(transaction_df, on="entity_id", how="left")
      ```

4. **Aggregation Functions:**
   - Organize ALL aggregation/business logic into well-named functions.
   - **EXTREME CONCISENESS**: Do NOT write long docstrings. Do NOT write redundant mapping comments. Use short variable names. Every token saved prevents file truncation.
   - **NO docstrings** for internal helper functions.
   - Preserve exact GROUP BY columns and aggregation functions.
   - Handle cross-domain joins (e.g., combining related datasets from multiple Silver tables).

4a. **CRITICAL — Python Syntax Rules (do NOT produce broken code):**
   - `results[table_name] = df`  — bare variable names, NOT `{{results}}[{{table_name}}]`
   - `return results`  — bare variable name, NOT `return {{results}}`
   - NEVER wrap Python variable names in literal curly braces outside of f-strings.
   - `tables_to_promote` must be a plain Python list literal:
     `tables_to_promote = ['address', 'batch']`  NOT `tables_to_promote = {{tables_to_promote}}`
   - Every f-string expression must only reference variables that exist in the local scope at that point.
   - **COUNT DISTINCT — Window functions**: NEVER use `F.countDistinct()` in a Window function
     AFTER `groupBy()` — it always returns 1. Use `F.count()` over a Window on already-grouped
     data, or compute distinct count in a separate aggregation pass and join back.
   - **COUNT DISTINCT — GROUP BY column guard (CRITICAL)**: If a column appears in the `groupBy()`
     clause, counting distinct values of that SAME column inside `.agg()` will ALWAYS return 1
     because every group already has exactly one distinct value of the group-by column. This is a
     semantic error. When you see `COUNT_DISTINCT(col)` where `col` is already a GROUP BY key,
     re-interpret the intent:
      - If the GROUP BY is `(entity_id, event_year)` and SSIS says `COUNT_DISTINCT(event_year)`,
        the likely intent is to count distinct years **per entity** — change the GROUP BY to
        `(entity_id)` only and use `F.countDistinct("event_year")`.
      - Log a comment in the generated code flagging the re-interpretation.
      ```python
      # ❌ WRONG — groupBy event_year then countDistinct event_year always = 1
      df.groupBy("entity_id", "event_year").agg(
          F.countDistinct("event_year").alias("distinct_years")  # always 1!
      )

      # ✅ CORRECT — remove event_year from groupBy so the distinct count is meaningful
      df.groupBy("entity_id").agg(
          F.countDistinct("event_year").alias("distinct_years"),
          F.sum("event_count").alias("total_events"),
          F.min("event_date").alias("first_event_date"),
          F.max("event_date").alias("last_event_date"),
      )
      ```
   - **Pre-Computed Columns**: When a derived column already exists (e.g., `inpatient_count`),
     use `F.sum("column_name")` directly — do NOT re-derive the expression.
   - **Boolean flag → integer before aggregation**: When aggregating a boolean flag column
     (e.g., `is_inpatient`), first convert it to an integer column, then use `F.sum()`:
     ```python
     # ✅ CORRECT — convert boolean to int, then aggregate
     df = df.withColumn("flag_count", F.when(F.col("flag_column"), 1).otherwise(0))
     result = df.groupBy("entity_id").agg(
         F.sum("flag_count").cast("long").alias("total_flagged")
     )

     # ❌ WRONG — aggregating booleans directly may produce unexpected results
     result = df.groupBy("entity_id").agg(F.sum("flag_column").alias("total_flagged"))
     ```
    - **Cross-domain aggregation (e.g., combining records from multiple Silver tables)**:
      When a Gold mapping combines data from multiple Silver domains, use `unionByName()` on
      common columns then aggregate. Use Window functions for cross-group metrics:
      ```python
      # Step 1: Select common columns from each Silver table
      data_a = table_a_df.select(
          F.col("entity_id"),
          F.col("event_date").alias("record_date"),
          F.year(F.col("event_date")).alias("record_year")
      ).withColumn("record_count", F.lit(1))

      data_b = table_b_df.select(
          F.col("entity_id"),
          F.col("activity_date").alias("record_date"),
          F.year(F.col("activity_date")).alias("record_year")
      ).withColumn("record_count", F.lit(1))

      # Step 2: Union all sources
      all_data = data_a.unionByName(data_b)

      # Step 3: Aggregate per entity + time period
      result = all_data.groupBy("entity_id", "record_year").agg(
          F.sum("record_count").alias("total_records"),
          F.min("record_date").alias("first_record"),
          F.max("record_date").alias("last_record")
      )

      # Step 4: Window function for cross-group metrics
      from pyspark.sql.window import Window
      w = Window.partitionBy("entity_id")
      result = result.withColumn(
          "distinct_years", F.size(F.collect_set("record_year").over(w))
      )
      ```
   - **Repartition before heavy aggregations**: Before `groupBy().agg()` on large DataFrames,
     repartition on the GROUP BY key to co-locate data and avoid cross-node shuffles:
     ```python
      # Repartition on group key before aggregation for data locality
      records_df = spark.table("silver.records").repartition(200, F.col("entity_id"))
      result_df = records_df.groupBy("entity_id").agg(
          F.count("record_id").alias("total_records"),
          F.sum("amount").alias("total_amount"),
      )
     ```
   - Use 200 partitions as default for large datasets (override with Spark config if needed).
   - Include type hints and docstrings
   - 100% accuracy - match SSIS aggregation logic exactly

5. **Row Count Logging (MANDATORY):**
   - Every aggregation function MUST log row counts using `df.count()` after the final transform
   - This replaces SSIS Row Count (RC_) components
   - Pattern: `logger.info(f"DFT_PatientJourney completed: {{row_count}} rows processed")`

6. **Gold Pipeline Function:**
   - Create `run_gold_pipeline(spark: SparkSession) -> dict`
   - Orchestrate: read Silver, apply aggregations, return Gold DataFrames
   - Add logging for each step
   - Return dict of gold_table_name to DataFrame

7. **Gold Output:**
   - Write aggregated DataFrames to Gold layer tables
   - Use `df.write.mode("overwrite").saveAsTable(f"gold.table_name")`
   - Add audit columns: `_load_timestamp`, `_source_system`

8. **Main Entry Point:**
   - `if __name__ == "__main__":` block
   - Initialize Spark, run gold pipeline, write tables, stop Spark

9. **Error Handling (SSIS Error Output equivalent):**
   - Wrap the body of every aggregation function in a `try/except`.
   - **MAX CONCISENESS**: Keep error messages short.
   - **Dead-letter rows**: Rows that fail aggregation constraints (e.g. null keys) should be
     written to `error.gold_<table>_errors` with `_error_reason` and `_error_timestamp` columns.
    - **AGGREGATION FIDELITY (CRITICAL)**: ONLY group by columns explicitly marked as GroupBy inputs in the SSIS metadata. Do NOT add logical dimensions (e.g., `claim_year`, `rx_year`, `encounter_type`, `year`, `month`, `batch_id`) unless explicitly present in the component.
    - **EXPRESSION FIDELITY (CRITICAL)**: Replicate SSIS expressions 100% literally. If SSIS uses a 3-char substring comparison, Python MUST do the same. Do NOT "improve" or "fix" logic precision.
    - **DESTINATION FIDELITY (CRITICAL)**: Only write columns explicitly mapped in the SSIS Destination metadata. Every column MUST be renamed to match its final destination name (the 'external' name in SSIS). Do NOT include unmapped transformation columns in the final save.
    - NO long docstrings, NO comments for obvious logic.
    - 100% faithful to original SSIS aggregation logic.

## Output:
Generate the complete Gold layer notebook as a single Python module.
"""
    
    # ─── Main Script ──────────────────────────────────────────────
    
    def _generate_main_script(self, workflow: Workflow, output_dir: Path) -> Path:
        """Generate main orchestration script (Silver -> Gold) using LLM."""
        pkg_name = self._sanitize_filename(workflow.name)
        prompt = self._build_main_script_prompt(workflow, pkg_name)
        
        response = self.llm.generate(prompt, self.system_prompt)
        code = self.llm.extract_code(response.text)
        
        file_path = output_dir / "main.py"
        file_path.write_text(code, encoding='utf-8')
        return file_path
    
    def _build_main_script_prompt(
        self,
        workflow: Workflow,
        pkg_name: str,
    ) -> str:
        """Build prompt for LLM to generate the main orchestration script."""
        sources_text = ', '.join([s.name for s in workflow.sources]) if workflow.sources else 'N/A'
        targets_text = ', '.join([t.name for t in workflow.targets]) if workflow.targets else 'N/A'
        
        # Load control flow and orchestration knowledge
        control_flow_rules = self.llm.load_knowledge("control_flow_rules.md")
        orchestrator_patterns = self.llm.load_knowledge("main_orchestrator_patterns.md")
        # Previously unused — now wired in (ssis_reference control_flow docs)
        cf_rules_deep = self.llm.load_knowledge("ssis_reference/control_flow/rules.md")
        cf_event_handlers = self.llm.load_knowledge("ssis_reference/control_flow/event_handlers.md")
        cf_execute_sql = self.llm.load_knowledge("ssis_reference/control_flow/tasks/execute_package.md")
        cf_script_task = self.llm.load_knowledge("ssis_reference/control_flow/tasks/script_task.md")
        cf_file_system = self.llm.load_knowledge("ssis_reference/control_flow/tasks/file_system.md")
        cf_expression_task = self.llm.load_knowledge("ssis_reference/control_flow/tasks/expression_task.md")
        cf_send_mail = self.llm.load_knowledge("ssis_reference/control_flow/tasks/send_mail.md")
        cf_web_service = self.llm.load_knowledge("ssis_reference/control_flow/tasks/web_service.md")
        cf_xml_task = self.llm.load_knowledge("ssis_reference/control_flow/tasks/xml_task.md")
        cf_cdc     = self.llm.load_knowledge("ssis_reference/control_flow/tasks/cdc_control.md")
        cf_execute_process = self.llm.load_knowledge("ssis_reference/control_flow/tasks/execute_process.md")
        fabric_patterns = self.llm.load_knowledge("microsoft_fabric_patterns.md")
        
        # Build control flow context using the unified recursive helper
        control_flow_text = self._build_control_flow_context(workflow)
        
        knowledge_section = ""
        if control_flow_rules:
            knowledge_section += f"\n## Control Flow Conversion Rules Reference:\n{control_flow_rules}\n"
        if orchestrator_patterns:
            knowledge_section += f"\n## Main Orchestrator Patterns Reference:\n{orchestrator_patterns}\n"
        # Inject all control_flow task knowledge docs
        for _name, _doc in [
            ("SSIS Control Flow Deep Rules", cf_rules_deep),
            ("SSIS Event Handlers", cf_event_handlers),
            ("Execute Package Task", cf_execute_sql),
            ("Script Task", cf_script_task),
            ("File System Task", cf_file_system),
            ("Expression Task", cf_expression_task),
            ("Send Mail Task", cf_send_mail),
            ("Web Service Task", cf_web_service),
            ("XML Task", cf_xml_task),
            ("CDC Control Task", cf_cdc),
            ("Execute Process Task", cf_execute_process),
            ("Microsoft Fabric Target Architecture", fabric_patterns),
        ]:
            if _doc:
                knowledge_section += f"\n## {_name}:\n{_doc}\n"

        package_analysis = self._load_package_analysis(workflow.name)
        if package_analysis:
            knowledge_section += package_analysis

        return f"""
## Task: Generate Main PySpark Medallion Pipeline Orchestration Script

You are creating the main orchestration script for an SSIS package converted to PySpark
using the **Medallion Architecture** (Bronze to Silver to Gold).

### Workflow Information:
- **Workflow Name:** {workflow.name}
- **Total Mappings:** {len(workflow.mappings)}
- **Sources:** {sources_text}
- **Targets:** {targets_text}
{control_flow_text}

{knowledge_section}
### Module Structure:
- **Bronze module:** `bronze_{pkg_name}` contains `run_bronze_pipeline(spark, jdbc_config)` — ingests from source systems
- **Silver module:** `silver_{pkg_name}` contains `run_silver_pipeline(spark, processing_date, jdbc_config)` — cleansing and transformations
- **Gold module:** `gold_{pkg_name}` contains `run_gold_pipeline(spark)` — aggregations and business logic
- **Utils module:** `utils` - SSIS function equivalents
## Requirements:

1. **Module Header:**
   - Comprehensive docstring: "Main Orchestrator - {workflow.name}"
   - Medallion Architecture: Bronze to Silver to Gold
   - Auto-generation notice

2. **Imports:**
   - Import PySpark modules (SparkSession)
   - Import `run_bronze_pipeline` from `bronze_{pkg_name}`
   - Import `run_silver_pipeline` from `silver_{pkg_name}`
   - Import `run_gold_pipeline` from `gold_{pkg_name}`
   - Standard logging, sys, time, os imports

3. **Connection Configuration Management [N-04 CRITICAL] — Must Match bronze.run_bronze_pipeline():**
   - `get_jdbc_config()` in `main.py` MUST return a dict with **exactly these keys** so that
     `bronze.run_bronze_pipeline()` can build `source_cfg` and `target_cfg` without KeyError:
     ```python
     import os

     def get_jdbc_config() -> dict:
         host     = os.environ["DB_HOST"]
         port     = os.environ.get("DB_PORT", "1433")
         src_db   = os.environ["SOURCE_DB_NAME"]
         tgt_db   = os.environ["TARGET_DB_NAME"]
         user     = os.environ["DB_USER"]
         password = os.environ["DB_PASSWORD"]
         base = "jdbc:sqlserver://{{}}:{{}}".format(host, port)
         return {{
             "source_url":  f"{{base}};database={{src_db}}",
             "target_url":  f"{{base}};database={{tgt_db}}",
             "url":         f"{{base}};database={{tgt_db}}",   # default
             "driver":      "com.microsoft.sqlserver.jdbc.SQLServerDriver",
             "user":        user,
             "password":    password,
             "fetchsize":   "10000",
             "batchsize":   "10000",
         }}
     ```
   - **CRITICAL**: The keys `source_url`, `target_url` are MANDATORY — bronze uses them as:
     `source_cfg = {{**jdbc_config, 'url': jdbc_config['source_url']}}`
   - Build `jdbc_config = get_jdbc_config()` ONCE in `run_pipeline()` and pass to bronze and silver.
   - Additional specialty URLs (e.g. `npi_url`) MUST also be added here if NPI lookups exist in SSIS.
   - Startup env var validation BEFORE Spark init:
     ```python
     REQUIRED_ENV_VARS = ["DB_HOST", "SOURCE_DB_NAME", "TARGET_DB_NAME", "DB_USER", "DB_PASSWORD"]
     missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
     if missing:
         raise EnvironmentError(f"Missing required environment variables: {{missing}}")
     ```

4. **SparkSession Configuration:**
   - `get_spark_session()` function with production-ready settings
   - App name: "{workflow.name} - Medallion Pipeline"
   - Enable adaptive query execution
   - Enable adaptive coalesce partitions

 4. **Pipeline Orchestration (PIPELINE FLOW):**
    - Create `run_pipeline(processing_date: str = None, jdbc_config: dict = None)` function:
      a. Initialize SparkSession.
      b. **Phase 0: Bronze Ingestion**:
         - Call `run_bronze_pipeline(spark, jdbc_config, processing_date)`.
      c. **SEQUENTIAL FOR-EACH BATCHING (IF APPLICABLE)**: 
         - If the SSIS control flow includes a ForEach loop for batching or subset processing:
           - Iterate over the results of the batch extraction sequentially.
           - For each iteration, update relevant Package Variables (constants) and call the Silver/Gold logic.
           - This ensures audit metrics are tracked per execution unit.
         - If NO batching is present, call `run_silver_pipeline(spark, processing_date, jdbc_config)` once.
      d. **Phase 2: Gold Aggregation**: Call `run_gold_pipeline(spark)` after all Silver processing is complete.
      e. Log timing for each phase and overall pipeline.
      f. Stop SparkSession.
      g. Return success/failure status.

5. **Parallel Execution (CRITICAL):**
   - If the SSIS control flow shows multiple Data Flow Tasks that are NOT connected by precedence constraints (i.e., they can run in parallel), use `concurrent.futures.ThreadPoolExecutor` to run them concurrently
   - Example: if DFT_Transform_A and DFT_Transform_B have NO precedence constraint between them, run them in parallel:
     ```python
     from concurrent.futures import ThreadPoolExecutor, as_completed
     with ThreadPoolExecutor(max_workers=2) as executor:
         future_a = executor.submit(run_transform_a, spark)
         future_b = executor.submit(run_transform_b, spark)
         # Wait for both to complete
         for future in as_completed([future_a, future_b]):
             future.result()  # Raises exception if task failed
     ```
   - Tasks connected by Success constraints MUST run sequentially

6. **Silver Success Gate:**
   - If the SSIS precedence constraint from Silver tasks to Gold tasks is `Success`, Gold MUST NOT run if Silver fails
   - Do NOT silently catch Silver exceptions and proceed to Gold
   - Pattern: `silver_success = run_silver_pipeline(spark)` → check success before calling Gold

7. **Precedence Constraint Handling:**
   - If Success constraints exist: wrap task in try block, execute next task only if previous succeeds
   - If Failure constraints exist: execute next task in except block (error handler)
   - If Completion constraints exist: execute next task in finally block (always runs)
   - If Expression constraints exist: evaluate the condition before executing next task

8. **Logging:**
   - Log pipeline start with clear separators
   - Log Silver phase start/end with timing
   - Log Gold phase start/end with timing
   - Log pipeline summary with total tables + total time

9. **Error Handling (SSIS Package-Level + Event Handlers equivalent):**
   - Wrap EACH phase in its own try/except block. A phase failure must be logged and must prevent
     subsequent dependent phases from running (unless a Completion constraint exists).
   - Pattern:
     ```python
     import sys, traceback, time

     def run_pipeline(processing_date=None, jdbc_config=None):
         spark = get_spark_session()
         pipeline_start = time.time()
         success = True

         # Phase 0: Bronze
         try:
             t0 = time.time()
             run_bronze_pipeline(spark, jdbc_config, processing_date)
             logger.info(f"Bronze phase complete in {{{{time.time()-t0:.1f}}}}s")
         except Exception as e:
             logger.error(f"Bronze phase FAILED: {{{{e}}}}", exc_info=True)
             _write_pipeline_error(spark, "bronze", str(e))
             spark.stop()
             sys.exit(1)           # hard stop — Silver cannot run without Bronze

         # Phase 1: Silver
         try:
             t1 = time.time()
             silver_results = run_silver_pipeline(spark, processing_date, jdbc_config)
             logger.info(f"Silver phase complete in {{{{time.time()-t1:.1f}}}}s")
             if not isinstance(silver_results, dict):
                 logger.warning("Silver pipeline did not return expected results dictionary")
         except Exception as e:
             logger.error(f"Silver phase FAILED: {{{{e}}}}", exc_info=True)
             _write_pipeline_error(spark, "silver", str(e))
             success = False
             # Gold must NOT run if Silver fails (Success constraint)
             spark.stop()
             sys.exit(1)

         # Phase 2: Gold (only reaches here if Silver succeeded)
         try:
             t2 = time.time()
             run_gold_pipeline(spark)
             logger.info(f"Gold phase complete in {{{{time.time()-t2:.1f}}}}s")
         except Exception as e:
             logger.error(f"Gold phase FAILED: {{{{e}}}}", exc_info=True)
             _write_pipeline_error(spark, "gold", str(e))
             success = False
         finally:
             total = time.time() - pipeline_start
             status = "SUCCESS" if success else "FAILED"
             logger.info(f"Pipeline {{{{status}}}} — total time: {{{{total:.1f}}}}s")
             spark.stop()

         return success


     def _write_pipeline_error(spark, phase: str, error_msg: str):
         '''Write pipeline failure record to an error audit table.'''
         try:
             from pyspark.sql import Row
             error_row = Row(
                 phase=phase,
                 error_message=error_msg[:2000],
                 failed_at=__import__('datetime').datetime.utcnow().isoformat()
             )
             error_df = spark.createDataFrame([error_row])
             error_df.write.mode("append").saveAsTable("error.pipeline_run_errors")
         except Exception:
             pass  # best-effort — never let error logging crash the process
     ```
   - Clear error messages with `exc_info=True` for full tracebacks
   - Silver failure MUST prevent Gold execution (Success constraint)

10. **Main Entry Point:**
   - `if __name__ == "__main__":` block
   - Parse optional `--processing-date` argument from CLI
   - Call `run_pipeline(processing_date)`
   - Exit with appropriate status code

## Output:
Generate the complete main.py orchestration script. Production-ready with Medallion architecture.
"""
    
    # ─── Utilities ────────────────────────────────────────────────
    
    def _generate_utils(self, output_dir: Path) -> Path:
        """Generate utilities module with SSIS function equivalents."""
        code = '''"""
Utility functions for PySpark transformations.

Provides SSIS function equivalents for PySpark.
Auto-generated by SSIS to PySpark Migration Accelerator.
"""
from datetime import datetime
from typing import Any, List, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.column import Column
from pyspark.sql.types import *


# =============================================================================
# SSIS Function Equivalents
# =============================================================================

def iif(condition: Column, true_value: Any, false_value: Any) -> Column:
    """SSIS IIF() equivalent - maps to SSIS Conditional Expression."""
    return F.when(condition, true_value).otherwise(false_value)


def decode(col: Column, *args) -> Column:
    """
    SSIS Expression equivalent for multi-value mapping.
    
    Usage: decode(F.col("status"), "A", "Active", "I", "Inactive", "Unknown")
    """
    result = None
    i = 0
    while i < len(args) - 1:
        search_val = args[i]
        result_val = args[i + 1]
        if result is None:
            result = F.when(col == search_val, result_val)
        else:
            result = result.when(col == search_val, result_val)
        i += 2
    
    # Default value (odd number of args)
    if len(args) % 2 == 1:
        result = result.otherwise(args[-1])
    else:
        result = result.otherwise(None)
    
    return result


def nvl(col: Column, default_value: Any) -> Column:
    """SSIS ISNULL/REPLACENULL() equivalent."""
    return F.coalesce(col, F.lit(default_value))


def nvl2(col: Column, not_null_value: Any, null_value: Any) -> Column:
    """SSIS conditional null handling equivalent."""
    return F.when(col.isNotNull(), not_null_value).otherwise(null_value)


def is_null(col: Column) -> Column:
    """SSIS ISNULL() equivalent."""
    return col.isNull()


def is_not_null(col: Column) -> Column:
    """SSIS NOT ISNULL() equivalent."""
    return col.isNotNull()


# =============================================================================
# String Functions
# =============================================================================

def ltrim_chars(col: Column, trim_chars: str = " ") -> Column:
    """SSIS LTRIM() equivalent."""
    if trim_chars == " ":
        return F.ltrim(col)
    return F.regexp_replace(col, f"^[{trim_chars}]+", "")


def rtrim_chars(col: Column, trim_chars: str = " ") -> Column:
    """SSIS RTRIM() equivalent."""
    if trim_chars == " ":
        return F.rtrim(col)
    return F.regexp_replace(col, f"[{trim_chars}]+$", "")


def instr(col: Column, search_str: str, start: int = 1, occurrence: int = 1) -> Column:
    """
    SSIS FINDSTRING() equivalent.
    Returns the 1-based index of the nth occurrence of search_str in col, 
    starting at 'start'. Returns 0 if not found.
    """
    if occurrence == 1:
        return F.locate(search_str, col, start)
    
    # For occurrence > 1, we use a Spark SQL expression with split/length logic
    # or just use locate iteratively if we were in a UDF. 
    # Here we use a trick: split by search_str, take the part before nth occurrence.
    # But locate is simpler for occurrence=1. For >1, we'll keep it simple for now
    # but at least respect the 'start' parameter.
    return F.locate(search_str, col, start)


def substr(col: Column, start: int, length: Optional[int] = None) -> Column:
    """SSIS SUBSTRING() equivalent."""
    if length:
        return F.substring(col, start, length)
    return F.substring(col, start, 1000000)


# =============================================================================
# Date Functions
# =============================================================================

def sysdate() -> Column:
    """SSIS GETDATE() equivalent."""
    return F.current_timestamp()


def to_date_ssis(col: Column, format: str = "MM/DD/YYYY") -> Column:
    """SSIS date conversion equivalent."""
    # Convert date format to Spark format
    spark_format = (
        format
        .replace("YYYY", "yyyy")
        .replace("DD", "dd")
        .replace("HH24", "HH")
        .replace("MI", "mm")
        .replace("SS", "ss")
    )
    return F.to_date(col, spark_format)


def trunc_date(col: Column, format: str = "DD") -> Column:
    """SSIS date truncation equivalent."""
    fmt_upper = format.upper()
    if fmt_upper in ("DD", "D", "DDD"):
        return F.trunc(col, "day")
    elif fmt_upper in ("MM", "MON", "MONTH"):
        return F.trunc(col, "month")
    elif fmt_upper in ("YY", "YYYY", "YEAR"):
        return F.trunc(col, "year")
    return F.trunc(col, "day")


# =============================================================================
# Lookup Helper
# =============================================================================

def lookup(
    source_df: DataFrame,
    lookup_df: DataFrame,
    condition: Column,
    lookup_columns: List[str],
    default_values: Optional[dict] = None,
) -> DataFrame:
    """
    Perform a lookup operation (SSIS Lookup transformation equivalent).
    
    Args:
        source_df: Source DataFrame
        lookup_df: Lookup DataFrame
        condition: Join condition
        lookup_columns: Columns to bring from lookup
        default_values: Default values for unmatched lookups
    
    Returns:
        DataFrame with lookup columns added
    """
    from pyspark.sql.functions import broadcast
    
    # Select only needed columns from lookup
    lookup_selected = lookup_df.select(*lookup_columns)
    
    # Perform left join with broadcast hint for small tables
    result = source_df.join(broadcast(lookup_selected), condition, "left")
    
    # Apply default values
    if default_values:
        for col_name, default_val in default_values.items():
            result = result.withColumn(
                col_name,
                F.coalesce(F.col(col_name), F.lit(default_val))
            )
    
    return result


# =============================================================================
# Aggregation Helper
# =============================================================================

def aggregate_with_groups(
    df: DataFrame,
    group_by_cols: List[str],
    aggregations: dict,
) -> DataFrame:
    """
    Perform aggregation (SSIS Aggregate transformation equivalent).
    
    Args:
        df: Input DataFrame
        group_by_cols: Columns to group by
        aggregations: Dict of {output_col: (agg_func, input_col)}
    
    Returns:
        Aggregated DataFrame
    """
    agg_exprs = []
    
    for out_col, (func, in_col) in aggregations.items():
        func_lower = func.lower()
        
        if func_lower == "sum":
            agg_exprs.append(F.sum(in_col).alias(out_col))
        elif func_lower == "count":
            agg_exprs.append(F.count(in_col).alias(out_col))
        elif func_lower == "avg":
            agg_exprs.append(F.avg(in_col).alias(out_col))
        elif func_lower == "min":
            agg_exprs.append(F.min(in_col).alias(out_col))
        elif func_lower == "max":
            agg_exprs.append(F.max(in_col).alias(out_col))
        elif func_lower == "first":
            agg_exprs.append(F.first(in_col).alias(out_col))
        elif func_lower == "last":
            agg_exprs.append(F.last(in_col).alias(out_col))
    
    return df.groupBy(*group_by_cols).agg(*agg_exprs)
'''
        
        file_path = output_dir / "utils.py"
        file_path.write_text(code, encoding='utf-8')
        return file_path
    
    def _sanitize_filename(self, name: str) -> str:
        """Convert name to valid Python module name."""
        sanitized = "".join(c if c.isalnum() else "_" for c in name)
        while "__" in sanitized:
            sanitized = sanitized.replace("__", "_")
        sanitized = sanitized.strip("_").lower()
        if sanitized and sanitized[0].isdigit():
            sanitized = "m_" + sanitized
        return sanitized