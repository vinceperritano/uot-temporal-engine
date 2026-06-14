"""
UOT Temporal Extrapolation Engine — Web API v1.1
==================================================
FastAPI backend wrapping uot_engine_v12.py.

Changes from v1.0 (based on GPT code review):
  1. construct_dataclass() helper — future-proof against new engine fields
  2. world_from_dict() uses construct_dataclass; fixes bool("false") bug;
     coerces time_estimate to float|None
  3. set_edges() now recomputes causal support / coherence / energy after
     topology change, and saves updated edges_json (not stale copy)
  4. seed_run() stores source_causal_evidence in node metadata, separating
     original source evidence from graph-derived causal support
  5. run_simulation() adds restart_from_reviewed param; persists step_history
  6. params_json renamed to settings_json throughout
  7. NodeEditRequest expanded with full set of editable UOT fields
  8. Removed unused BackgroundTasks and StaticFiles imports
  9. CORS note added for pre-deployment tightening

Run with:
    pip install fastapi uvicorn pydantic
    uvicorn uot_api_v2:app --reload --port 8000
"""

from __future__ import annotations

import dataclasses
import json
import os
import math
import sqlite3
import sys
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))

from uot_engine_v12_patched import (
    ModelParams, WorldTimelineState, TemporalState,
    EventGraph, EventNode, EventEdge,
    ObserverState, EvidenceEvent,
    SourcePacket, SeededEvent, ExtractionDiagnostics,
    estimate_uot_fields,
    build_seeded_test_world_v12,
    collect_and_extract_seeded_events,
    generate_search_queries,
    web_search_sources, format_source_packets_for_stage_a,
    infer_causal_edges_from_candidates,
    infer_branch_groups_and_causal_candidates_specialized,
    apply_structure_to_event_candidates,
    build_graph_from_estimates,
    recompute_coherence_and_energy,
    simulation_step, compute_instability_score,
    clamp, normalize_branch_groups,
    apply_pending_evidence,
    LIVE_MODE,
    # Horizon
    HorizonConfig, steps_from_horizon,
    # Iterative loop
    run_iterative_simulation, detect_discrepancies, IterationResult,
    compute_resolution_state,
    # Shared utilities
    call_anthropic_api,
)


# ══════════════════════════════════════════════════════════════════════════════
# Rate limiting (Phase 4)
# ══════════════════════════════════════════════════════════════════════════════
MAX_RUNS_PER_IP_PER_HOUR = int(os.environ.get("MAX_RUNS_PER_IP_PER_HOUR", "20"))
MAX_RUNS_PER_IP_PER_DAY  = int(os.environ.get("MAX_RUNS_PER_IP_PER_DAY",  "100"))

RATE_LIMIT_BYPASS_KEY = os.environ.get("RATE_LIMIT_BYPASS_KEY", "")

def _check_rate_limit(conn, ip: str, bypass_key: str = "") -> tuple:
    """Returns (allowed: bool, message: str). Checks live-run rate limits."""
    if not LIVE_MODE:
        return True, ""
    # Allow bypass for trusted users with the secret key
    if RATE_LIMIT_BYPASS_KEY and bypass_key == RATE_LIMIT_BYPASS_KEY:
        return True, ""
    now = datetime.utcnow()
    hour_ago = (now - timedelta(hours=1)).isoformat()
    day_ago  = (now - timedelta(hours=24)).isoformat()
    hourly = conn.execute(
        "SELECT COUNT(*) FROM rate_limits WHERE ip=? AND created_at>?", (ip, hour_ago)
    ).fetchone()[0]
    daily = conn.execute(
        "SELECT COUNT(*) FROM rate_limits WHERE ip=? AND created_at>?", (ip, day_ago)
    ).fetchone()[0]
    if hourly >= MAX_RUNS_PER_IP_PER_HOUR:
        return False, (f"Rate limit: {MAX_RUNS_PER_IP_PER_HOUR} live runs per hour maximum. "
                       f"Please wait a bit before starting another run.")
    if daily >= MAX_RUNS_PER_IP_PER_DAY:
        return False, (f"Rate limit: {MAX_RUNS_PER_IP_PER_DAY} live runs per day maximum. "
                       f"Please try again later.")
    return True, ""


# ══════════════════════════════════════════════════════════════════════════════
# Database
# ══════════════════════════════════════════════════════════════════════════════

# UOT_DB_PATH env var lets Railway/Render mount a persistent volume
DB_PATH = Path(os.getenv("UOT_DB_PATH", str(Path(__file__).parent / "uot_runs.db")))


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            id             TEXT PRIMARY KEY,
            topic          TEXT NOT NULL,
            observer_basis TEXT NOT NULL,
            status         TEXT NOT NULL DEFAULT 'created',
            settings_json  TEXT,
            series_resonance_json TEXT,
            created_at     TEXT NOT NULL,
            updated_at     TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS world_states (
            id          TEXT PRIMARY KEY,
            run_id      TEXT NOT NULL,
            stage       TEXT NOT NULL,
            world_json  TEXT NOT NULL,
            scores_json TEXT,
            edges_json  TEXT,
            created_at  TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS run_artifacts (
            id                       TEXT PRIMARY KEY,
            run_id                   TEXT NOT NULL,
            initial_field_json       TEXT,
            reviewed_field_json      TEXT,
            final_field_json         TEXT,
            observer_basis_json      TEXT,
            convergence_summary_json TEXT,
            primary_question_json    TEXT,
            slot_assignments_json    TEXT,
            slot_influences_json     TEXT,
            created_at               TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS iteration_logs (
            id                      TEXT PRIMARY KEY,
            run_id                  TEXT NOT NULL,
            attempt_index           INTEGER NOT NULL DEFAULT 1,
            iteration_index         INTEGER NOT NULL,
            scores_before_json      TEXT,
            scores_after_json       TEXT,
            discrepancies_json      TEXT,
            near_discrepancies_json TEXT,
            corrections_json        TEXT,
            convergence_delta_json  TEXT,
            fallback_used           INTEGER DEFAULT 0,
            loop_error              TEXT,
            created_at              TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS rate_limits (
            id          TEXT PRIMARY KEY,
            ip          TEXT NOT NULL,
            created_at  TEXT NOT NULL
        );

        """)

        # ── Phase 4 schema migrations — safe to run every startup ─────────────
        for _sql in [
            "ALTER TABLE iteration_logs ADD COLUMN attempt_index INTEGER NOT NULL DEFAULT 1",
            "ALTER TABLE run_artifacts  ADD COLUMN primary_question_json TEXT",
        ]:
            try:
                conn.execute(_sql)
            except Exception:
                pass   # column already exists

        # ── Phase 6 schema migrations ─────────────────────────────────────────────
        for _sql in [
            "ALTER TABLE runs ADD COLUMN series_id        TEXT",
            "ALTER TABLE runs ADD COLUMN run_index        INTEGER",
            "ALTER TABLE runs ADD COLUMN trigger_type     TEXT DEFAULT 'manual'",
            "ALTER TABLE runs ADD COLUMN previous_run_id  TEXT",
        ]:
            try:   conn.execute(_sql)
            except Exception: pass

        conn.executescript("""
        CREATE TABLE IF NOT EXISTS scenario_series (
            series_id              TEXT PRIMARY KEY,
            title                  TEXT NOT NULL,
            original_question      TEXT NOT NULL,
            normalized_question    TEXT,
            horizon_config_json    TEXT,
            observer_basis_json    TEXT,
            primary_question_json  TEXT,
            canonical_slots_json   TEXT,
            created_at             TEXT NOT NULL,
            last_updated_at        TEXT NOT NULL,
            status                 TEXT NOT NULL DEFAULT 'active',
            refresh_policy         TEXT NOT NULL DEFAULT 'manual'
        );

        CREATE TABLE IF NOT EXISTS slot_snapshots (
            id                TEXT PRIMARY KEY,
            series_id         TEXT NOT NULL,
            run_id            TEXT NOT NULL,
            slot_id           TEXT NOT NULL,
            slot_label        TEXT,
            slot_polarity     TEXT,
            slot_kind         TEXT,
            probability       REAL,
            temporal_energy   REAL,
            temporal_coherence REAL,
            temporal_entropy  REAL,
            indeterminacy     REAL,
            assigned_event_id TEXT,
            event_label       TEXT,
            synthetic         INTEGER DEFAULT 0,
            resolved_status   TEXT,
            record_confidence REAL,
            created_at        TEXT NOT NULL,
            FOREIGN KEY (series_id) REFERENCES scenario_series(series_id),
            FOREIGN KEY (run_id)    REFERENCES runs(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS field_deltas (
            id               TEXT PRIMARY KEY,
            series_id        TEXT NOT NULL,
            from_run_id      TEXT NOT NULL,
            to_run_id        TEXT NOT NULL,
            slot_id          TEXT NOT NULL,
            probability_delta REAL,
            energy_delta     REAL,
            indeterminacy_delta REAL,
            resolution_change TEXT,
            summary          TEXT,
            resonance_score  REAL,
            created_at       TEXT NOT NULL
        );
        """)

        # Phase 7.1/7.4: migrate existing DBs to add new columns
        for _col, _tbl, _def in [
            ("slot_assignments_json", "run_artifacts", "TEXT"),
            ("slot_influences_json",  "run_artifacts", "TEXT"),
            ("series_resonance_json", "runs",          "TEXT"),
            ("resonance_score",       "field_deltas",  "REAL"),
            ("slot_label",            "slot_snapshots", "TEXT"),
        ]:
            try:
                conn.execute(f"ALTER TABLE {_tbl} ADD COLUMN {_col} {_def}")
                conn.commit()
            except Exception:
                pass  # column already exists

        conn.executescript("""
        CREATE TABLE IF NOT EXISTS observer_interventions (
            id                TEXT PRIMARY KEY,
            run_id            TEXT NOT NULL,
            intervention_type TEXT NOT NULL,
            target_type       TEXT,
            target_id         TEXT,
            before_json       TEXT,
            after_json        TEXT,
            timestamp         TEXT NOT NULL
        );

                CREATE TABLE IF NOT EXISTS audit_events (
            id          TEXT PRIMARY KEY,
            run_id      TEXT NOT NULL,
            event_type  TEXT NOT NULL,
            details     TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
        );
        """)


# ══════════════════════════════════════════════════════════════════════════════
# Serialization helpers
# ══════════════════════════════════════════════════════════════════════════════

def _to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses and containers to plain Python objects."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return 0.0
    return obj


def world_to_json(world: WorldTimelineState) -> str:
    return json.dumps(_to_dict(world))


# ── Fix 1: construct_dataclass — forwards-compatible reconstruction ────────────

def construct_dataclass(cls, data: dict, overrides: dict = None):
    """
    Safely construct a dataclass from a dict, ignoring unknown keys
    and applying only fields the dataclass actually accepts.

    This means old saved runs will not break when new engine versions
    add fields, and new saved runs will not break old API code.
    """
    allowed = {f.name for f in dataclasses.fields(cls)}
    merged  = {**data, **(overrides or {})}   # overrides win over raw JSON
    kwargs  = {k: v for k, v in merged.items() if k in allowed}
    return cls(**kwargs)


def _coerce_float_or_none(val) -> Optional[float]:
    """Return float or None; safe against strings and None."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def world_from_dict(data: dict) -> WorldTimelineState:
    """
    Reconstruct WorldTimelineState from a plain dict.

    Uses construct_dataclass() for forward-compatibility.
    Applies explicit numeric coercions for safety.
    """

    # ── EventGraph ────────────────────────────────────────────────────────────
    graph_data = data.get("event_graph", {})
    graph = EventGraph()

    for nid, nd in graph_data.get("nodes", {}).items():
        # Explicit coercions for numeric fields the engine computes on
        node_overrides = {
            "id":                 str(nd.get("id", nid)),
            "label":              str(nd.get("label", "")),
            "probability":        clamp(float(nd.get("probability", 0.5))),
            "time_estimate":      _coerce_float_or_none(nd.get("time_estimate")),
            "time_uncertainty":   clamp(float(nd.get("time_uncertainty", 0.5))),
            "temporal_energy":    clamp(float(nd.get("temporal_energy", 0.5))),
            "temporal_coherence": clamp(float(nd.get("temporal_coherence", 0.5))),
            "temporal_entropy":   clamp(float(nd.get("temporal_entropy", 0.5))),
            "indeterminacy":      clamp(float(nd.get("indeterminacy", 0.5))),
            "observer_sensitivity": clamp(float(nd.get("observer_sensitivity", 0.5))),
            "disruption_score":   clamp(float(nd.get("disruption_score", 0.5))),
            "record_coherence":   clamp(float(nd.get("record_coherence", 0.5))),
            "source_count":       int(nd.get("source_count", 0)),
            "source_agreement":   clamp(float(nd.get("source_agreement", 0.5))),
            # Dicts: preserved opaque — version-safe
            "categories":     nd.get("categories", {}),
            "auto_estimated": nd.get("auto_estimated", {}),
            "user_adjusted":  nd.get("user_adjusted", {}),
            "metadata":       nd.get("metadata", {}),
            "outcome_role":   nd.get("outcome_role", "causal_context"),
        }
        graph.nodes[nid] = construct_dataclass(EventNode, nd, node_overrides)

    for ed in graph_data.get("edges", []):
        edge_overrides = {
            "causal_weight":     clamp(float(ed.get("causal_weight", 0.5))),
            "delay":             float(ed.get("delay", 1.0)),
            "uncertainty":       clamp(float(ed.get("uncertainty", 0.5))),
            "feedback_strength": clamp(float(ed.get("feedback_strength", 0.0))),
        }
        graph.edges.append(construct_dataclass(EventEdge, ed, edge_overrides))

    # ── Observers ─────────────────────────────────────────────────────────────
    observers = []
    for od in data.get("observers", []):
        obs_overrides = {
            "coherence_level":   clamp(float(od.get("coherence_level", 0.5))),
            "coupling_strength": clamp(float(od.get("coupling_strength", 0.5))),
        }
        observers.append(construct_dataclass(ObserverState, od, obs_overrides))

    # ── Evidence Events ───────────────────────────────────────────────────────
    evidence = []
    for ev in data.get("evidence_events", []):
        ev_overrides = {
            "likelihood_ratio": float(ev.get("likelihood_ratio", 1.0)),
            "confidence":       clamp(float(ev.get("confidence", 1.0))),
            # Fix: bool("false") == True; use explicit comparison
            "applied": ev.get("applied") is True or ev.get("applied") == "true",
            "timestamp": _coerce_float_or_none(ev.get("timestamp")),
        }
        evidence.append(construct_dataclass(EvidenceEvent, ev, ev_overrides))

    # ── TemporalState ─────────────────────────────────────────────────────────
    ts_data = data.get("temporal_state", {})
    ts_overrides = {k: float(v) for k, v in ts_data.items()
                   if isinstance(v, (int, float, str))
                   and k in {f.name for f in dataclasses.fields(TemporalState)}}
    ts = construct_dataclass(TemporalState, ts_data, ts_overrides)

    return WorldTimelineState(
        event_graph=graph,
        observers=observers,
        evidence_events=evidence,
        temporal_state=ts,
        step=int(data.get("step", 0)),
    )


def world_from_json(s: str) -> WorldTimelineState:
    return world_from_dict(json.loads(s))


# ── Fix 2: recompute causal fields after topology change ──────────────────────

def _recompute_after_edge_change(world: WorldTimelineState,
                                  observer_basis: dict) -> WorldTimelineState:
    """
    After any change to graph edges, recompute graph-derived causal support,
    temporal coherence, and temporal energy for all nodes.

    Uses source_causal_evidence stored in node.metadata during seeding,
    NOT the previously-computed causal_support_graph, to avoid feedback loop.
    """
    seeded_causal_map = {
        nid: float(node.metadata.get("source_causal_evidence", 0.5))
        for nid, node in world.event_graph.nodes.items()
    }
    world.event_graph = recompute_coherence_and_energy(
        world.event_graph, observer_basis, seeded_causal_map
    )
    return world


# ══════════════════════════════════════════════════════════════════════════════
# Audit helpers
# ══════════════════════════════════════════════════════════════════════════════

def record_audit(conn: sqlite3.Connection, run_id: str,
                 event_type: str, details: dict) -> None:
    conn.execute(
        "INSERT INTO audit_events VALUES (?,?,?,?,?)",
        (str(uuid.uuid4()), run_id, event_type,
         json.dumps(details), datetime.now(timezone.utc).isoformat())
    )


def touch_run(conn: sqlite3.Connection, run_id: str, status: str) -> None:
    conn.execute(
        "UPDATE runs SET status=?, updated_at=? WHERE id=?",
        (status, datetime.now(timezone.utc).isoformat(), run_id)
    )


def _upsert_run_artifact(conn, run_id: str, **fields) -> None:
    """
    Upsert a run_artifacts row. Creates the row if it doesn't exist,
    otherwise updates only the specified fields. This lets seeding,
    simulation-start, and simulation-end each write their snapshot
    without overwriting each other.
    """
    row = conn.execute(
        "SELECT id FROM run_artifacts WHERE run_id = ? LIMIT 1", (run_id,)
    ).fetchone()
    now = datetime.now(timezone.utc).isoformat()
    if row:
        if fields:
            set_clause = ", ".join(f"{k} = ?" for k in fields)
            conn.execute(
                f"UPDATE run_artifacts SET {set_clause} WHERE run_id = ?",
                list(fields.values()) + [run_id]
            )
    else:
        all_fields = {"id": str(uuid.uuid4()), "run_id": run_id,
                      "created_at": now, **fields}
        cols = ", ".join(all_fields.keys())
        placeholders = ", ".join("?" * len(all_fields))
        conn.execute(
            f"INSERT INTO run_artifacts ({cols}) VALUES ({placeholders})",
            list(all_fields.values())
        )


def _get_state_for_stage(conn: sqlite3.Connection, run_id: str,
                          stage: Optional[str] = None) -> Optional[sqlite3.Row]:
    """
    Retrieve world state. If stage is given, get latest row for that stage.
    Otherwise get the latest row overall.
    """
    if stage:
        return conn.execute(
            "SELECT * FROM world_states WHERE run_id=? AND stage=? "
            "ORDER BY created_at DESC LIMIT 1",
            (run_id, stage)
        ).fetchone()
    return conn.execute(
        "SELECT * FROM world_states WHERE run_id=? ORDER BY created_at DESC LIMIT 1",
        (run_id,)
    ).fetchone()


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic models
# ══════════════════════════════════════════════════════════════════════════════

class CreateRunRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=300)
    observer_basis: Dict[str, float] = Field(default_factory=lambda: {
        "democratic_institutions": 0.7,
        "geopolitical_alliances":  0.7,
        "social_cohesion":         0.5,
        "economic_stability":      0.5,
    })
    demo_mode:      bool = Field(default=False)
    horizon_label:  str  = Field(default="6mo",
        description="Time horizon: 3mo | 6mo | 1yr | 2yr | 5yr | custom")
    horizon_custom: Optional[str] = Field(default=None,
        description="Custom horizon description, used if horizon_label='custom'")
    # Phase 6: pre-link to series at creation (eliminates separate link-series call)
    series_id: Optional[str] = None
    run_index: Optional[int] = None


class RunSummary(BaseModel):
    id: str
    topic: str
    observer_basis: Dict[str, float]
    status: str
    created_at: str
    updated_at: str


# Fix 3: Expanded NodeEditRequest with full set of observer-editable UOT fields
class NodeEditRequest(BaseModel):
    # Simulation field values
    probability:          Optional[float] = None
    temporal_energy:      Optional[float] = None
    temporal_coherence:   Optional[float] = None
    temporal_entropy:     Optional[float] = None
    indeterminacy:        Optional[float] = None
    # UOT structural / interpretive fields
    record_coherence:     Optional[float] = None
    disruption_score:     Optional[float] = None
    observer_sensitivity: Optional[float] = None
    source_agreement:     Optional[float] = None
    # Classification fields
    temporal_status:      Optional[str]   = None
    label:                Optional[str]   = None
    branch_group:         Optional[str]   = None
    branch_label:         Optional[str]   = None
    # Category weights (replaces the whole dict if provided)
    categories:           Optional[Dict[str, float]] = None


class EdgeEditRequest(BaseModel):
    source_id:        str
    target_id:        str
    relation_type:    str   = "causal"
    causal_weight:    float = 0.5
    uncertainty:      float = 0.5
    feedback_strength: float = 0.0


class AddEvidenceRequest(BaseModel):
    target_node_id:  str
    likelihood_ratio: float = Field(..., ge=0.01, le=10.0)
    confidence:      float  = Field(default=0.8, ge=0.0, le=1.0)
    source:          Optional[str] = None
    description:     str = ""


# Fix 4: SimulateRequest gains restart_from_reviewed
class BasisSuggestRequest(BaseModel):
    topic:         str  = Field(..., min_length=3)
    horizon_label: str  = Field(default="6mo")
    horizon_custom: Optional[str] = None


class BasisDimension(BaseModel):
    key:         str
    label:       str
    description: str
    weight:      float = 0.7


class BasisSuggestResponse(BaseModel):
    topic:  str
    basis:  List[BasisDimension]


class SimulateRequest(BaseModel):
    steps:                 int   = Field(default=10, ge=1, le=100)
    dt:                    float = Field(default=0.1, ge=0.01, le=1.0)
    max_iterations:        int   = Field(default=5, ge=1, le=10)
    use_iterative_loop:    bool  = Field(default=True)
    restart_from_reviewed: bool  = Field(
        default=True,
        description=(
            "True (default): simulate from latest reviewed/seeded state — "
            "a fresh run from the observer-confirmed graph. "
            "False: continue from latest simulated state."
        )
    )


# ══════════════════════════════════════════════════════════════════════════════
# Response helpers
# ══════════════════════════════════════════════════════════════════════════════

def _graph_to_response(world: WorldTimelineState,
                        scores: Optional[dict] = None) -> dict:
    graph = world.event_graph
    ts    = world.temporal_state

    nodes = []
    for nid, node in graph.nodes.items():
        nodes.append({
            "id":                  node.id,
            "label":               node.label,
            "probability":         round(node.probability, 4),
            "temporal_status":     node.temporal_status,
            "temporal_energy":     round(node.temporal_energy, 4),
            "temporal_coherence":  round(node.temporal_coherence, 4),
            "temporal_entropy":    round(node.temporal_entropy, 4),
            "indeterminacy":       round(node.indeterminacy, 4),
            "record_coherence":    round(node.record_coherence, 4),
            "observer_sensitivity":round(node.observer_sensitivity, 4),
            "disruption_score":    round(node.disruption_score, 4),
            "source_agreement":    round(node.source_agreement, 4),
            "categories":          node.categories,
            "branch_group":        node.branch_group,
            "branch_label":        node.branch_label,
            "canonical_slot_id":   getattr(node, "canonical_slot_id", None),
            "branch_probability":  (round(node.branch_probability, 4)
                                     if getattr(node, "branch_probability", None) is not None else None),
            "slot_raw_probability": (round(node.slot_raw_probability, 4)
                                      if getattr(node, "slot_raw_probability", None) is not None else None),
            "metadata":            getattr(node, "metadata", {}) or {},
            "outcome_role":        getattr(node, "outcome_role", "causal_context"),
            "source_count":        node.source_count,
            "confidence_note":     node.confidence_note,
            "auto_estimated":      node.auto_estimated,
            "user_adjusted":       node.user_adjusted,
            "instability_score": (
                round(scores["node_scores"].get(nid, 0.0), 4)
                if scores else None
            ),
        })

    edges = [
        {
            "source_id":       e.source_id,
            "target_id":       e.target_id,
            "relation_type":   e.relation_type,
            "causal_weight":   round(e.causal_weight, 4),
            "uncertainty":     round(e.uncertainty, 4),
            "feedback_strength": round(e.feedback_strength, 4),
        }
        for e in graph.edges
    ]

    result = {
        "nodes":  nodes,
        "edges":  edges,
        "temporal_state": {
            k: round(float(getattr(ts, k)), 4)
            for k in [f.name for f in dataclasses.fields(TemporalState)]
        },
        "step": world.step,
    }

    if scores:
        result["scores"] = {
            "global_instability": scores["global_instability"],
            "causal_conflict":    scores["causal_conflict"],
            "lambda_temporal":    scores["lambda_temporal"],
            "field_stability":    scores["field_stability"],
            "branch_details":     scores["branch_details"],
        }

    return result


# ══════════════════════════════════════════════════════════════════════════════
# App
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="UOT Temporal Extrapolation Engine",
    description="Observer-aware causal timeline modeling — Unified Observer Theory.",
    version="1.1.0",
)

# FRONTEND_ORIGIN: set to your deployed frontend URL in production.
# Defaults to "*" for local development.
# Example: export FRONTEND_ORIGIN="https://your-app.railway.app"
_FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")
_CORS_ORIGINS = ["*"] if _FRONTEND_ORIGIN == "*" else [_FRONTEND_ORIGIN, "http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()


# ══════════════════════════════════════════════════════════════════════════════
# Runs
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/runs", response_model=RunSummary, status_code=201)
def create_run(req: CreateRunRequest, request: Request = None):
    # ── Rate limiting (Phase 4) ────────────────────────────────────────────────
    if not req.demo_mode and LIVE_MODE:
        client_ip = "unknown"
        if request:
            client_ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
            client_ip = client_ip.split(",")[0].strip()
        bypass_key = request.headers.get("X-Rate-Limit-Bypass", "") if request else ""
        with get_db() as conn:
            allowed, msg = _check_rate_limit(conn, client_ip, bypass_key)
            if not allowed:
                raise HTTPException(status_code=429, detail=msg)
            # Record this run attempt
            conn.execute(
                "INSERT INTO rate_limits VALUES (?,?,?)",
                (str(uuid.uuid4()), client_ip, datetime.utcnow().isoformat())
            )

    run_id = str(uuid.uuid4())
    now    = datetime.now(timezone.utc).isoformat()
    # Fix 5: renamed params_json → settings_json
    settings = json.dumps({
        "demo_mode":      req.demo_mode,
        "horizon_label":  req.horizon_label,
        "horizon_custom": req.horizon_custom,
    })
    with get_db() as conn:
        conn.execute(
            """INSERT INTO runs
               (id, topic, observer_basis, status, settings_json,
                created_at, updated_at,
                series_id, run_index, trigger_type, previous_run_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (run_id, req.topic, json.dumps(req.observer_basis),
             "created", settings, now, now,
             req.series_id, req.run_index,
             "manual_reseed" if req.series_id else "manual",
             None)
        )
        record_audit(conn, run_id, "run_created", {
            "topic": req.topic,
            "observer_basis": req.observer_basis,
            "demo_mode": req.demo_mode,
        })
    return RunSummary(id=run_id, topic=req.topic,
                      observer_basis=req.observer_basis,
                      status="created", created_at=now, updated_at=now)


@app.get("/api/runs")   # no response_model — returns plain dicts with series_id
def list_runs():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC"
        ).fetchall()
    result = []
    for r in rows:
        try:   ob = json.loads(r["observer_basis"] or "{}")
        except: ob = {}
        result.append({
            "id":             r["id"],
            "topic":          r["topic"],
            "observer_basis": ob,
            "status":         r["status"],
            "series_id":      r["series_id"],
            "created_at":     r["created_at"],
            "updated_at":     r["updated_at"],
        })
    return result


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Run not found")
        state_row = _get_state_for_stage(conn, run_id)

    result = {
        "id":            row["id"],
        "topic":         row["topic"],
        "observer_basis":json.loads(row["observer_basis"]),
        "status":        row["status"],
        "created_at":    row["created_at"],
        "updated_at":    row["updated_at"],
        "series_id":     row["series_id"],
        "graph":         None,
        "scores":        None,
    }
    if state_row:
        world  = world_from_json(state_row["world_json"])
        scores = json.loads(state_row["scores_json"]) if state_row["scores_json"] else None
        result["graph"] = _graph_to_response(world, scores)
    return result


@app.delete("/api/runs/{run_id}", status_code=204)
def delete_run(run_id: str):
    with get_db() as conn:
        conn.execute("DELETE FROM runs WHERE id=?", (run_id,))


# ══════════════════════════════════════════════════════════════════════════════
# Seeding pipeline
# ══════════════════════════════════════════════════════════════════════════════

def _run_seed_pipeline(run_id: str) -> None:
    """
    Background task: full seeding pipeline.
    Each stage has individual error handling so partial failures produce
    a degraded-but-usable graph rather than a total failure.
    """
    stage = "init"
    try:
        with get_db() as conn:
            run = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if not run:
                return

        settings       = json.loads(run["settings_json"] or "{}")
        demo_mode      = settings.get("demo_mode", False)
        observer_basis = json.loads(run["observer_basis"])

        # Build HorizonConfig from stored settings
        hl = settings.get("horizon_label", "6mo")
        hc = settings.get("horizon_custom")
        if hl == "custom" and hc:
            import datetime as _dt
            today = _dt.date.today()
            horizon = HorizonConfig(
                start_date=today, target_date=today + _dt.timedelta(days=365),
                horizon_days=365, horizon_label=hc
            )
        else:
            horizon = HorizonConfig.from_label(hl)

        if demo_mode:
            stage = "demo_build"
            world         = build_seeded_test_world_v12()
            stage_c_edges = []
        else:
            # ── Stage 0+0.5+A+B+C+D: live pipeline ──────────────────────────
            stage = "collect_events"
            # Phase 6: load locked canonical slots BEFORE the try block (Python 3.12 scoping)
            _locked_pq    = None
            _locked_slots = None
            _run_series_id = dict(run).get("series_id")
            if _run_series_id:
                try:
                    _srow = conn.execute(
                        "SELECT primary_question_json, canonical_slots_json"
                        " FROM scenario_series WHERE series_id=?",
                        (_run_series_id,)
                    ).fetchone()
                    if not _srow or not _srow["canonical_slots_json"]:
                        print(f"[API] Series {_run_series_id}: no canonical_slots_json — Stage Q will run fresh")
                    if _srow and _srow["canonical_slots_json"]:
                        _locked_pq    = json.loads(_srow["primary_question_json"] or "{}")
                        _locked_slots = json.loads(_srow["canonical_slots_json"] or "[]")
                        print(f"[API] Re-observation: {len(_locked_slots)} locked slots")
                except Exception as _le:
                    print(f"[API] Could not load locked slots: {_le}")
            try:
                result = collect_and_extract_seeded_events(
                    run["topic"], observer_basis, horizon,
                    locked_primary_question=_locked_pq,
                    locked_canonical_slots=_locked_slots,
                )
                if isinstance(result, tuple) and len(result) >= 4:
                    seeded_events, stage_c_edges, _slot_assignments, _slot_influences = result
                elif isinstance(result, tuple) and len(result) == 2:
                    seeded_events, stage_c_edges = result
                    _slot_assignments, _slot_influences = [], []
                else:
                    seeded_events, stage_c_edges = result, []
                    _slot_assignments, _slot_influences = [], []
            except Exception as e:
                # Record stage error but continue with empty event list
                with get_db() as conn:
                    record_audit(conn, run_id, "stage_error", {"stage": stage, "error": str(e)[:300]})
                raise RuntimeError(f"Stage A/B/C failed: {e}") from e

            if not seeded_events:
                raise RuntimeError(
                    "Pipeline returned no events. Check that TAVILY_API_KEY and "
                    "ANTHROPIC_API_KEY are set correctly in Railway environment variables."
                )

            stage = "build_graph"
            estimated_nodes   = [estimate_uot_fields(se, observer_basis) for se in seeded_events]
            final_edges       = stage_c_edges or infer_causal_edges_from_candidates(seeded_events)
            seeded_causal_map = {se.id: se.causal_support for se in seeded_events}

            graph = build_graph_from_estimates(estimated_nodes, final_edges)

            stage = "recompute"
            graph = recompute_coherence_and_energy(graph, observer_basis, seeded_causal_map)

            for nid, node in graph.nodes.items():
                node.metadata.setdefault(
                    "source_causal_evidence",
                    str(seeded_causal_map.get(nid, 0.5))
                )

            observer = ObserverState(
                id="observer_1", label="User observer",
                measurement_basis=observer_basis,
                coupling_strength=0.6, coherence_level=0.7,
            )
            world = WorldTimelineState(
                event_graph=graph, observers=[observer],
                temporal_state=TemporalState(),
            )

        # Demo fallback backfill
        DEMO_SOURCE_CAUSAL = {
            "e1": 0.90, "e2": 0.85, "e3a": 0.70, "e3b": 0.55, "e3c": 0.30,
            "e4": 0.75, "e5": 0.60, "e6a": 0.55, "e6b": 0.45, "e6c": 0.40, "e7": 0.60,
        }
        for nid, node in world.event_graph.nodes.items():
            node.metadata.setdefault(
                "source_causal_evidence",
                str(DEMO_SOURCE_CAUSAL.get(nid, 0.5))
            )

        stage = "score"
        params = ModelParams()
        scores = compute_instability_score(world, params)
        now    = datetime.now(timezone.utc).isoformat()

        with get_db() as conn:
            conn.execute(
                "INSERT INTO world_states VALUES (?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), run_id, "seeded",
                 world_to_json(world), json.dumps(scores),
                 json.dumps(stage_c_edges), now)
            )
            touch_run(conn, run_id, "seeded")
            record_audit(conn, run_id, "pipeline_seeded", {
                "demo_mode":     demo_mode,
                "node_count":    len(world.event_graph.nodes),
                "edge_count":    len(world.event_graph.edges),
            })
            # Save initial field snapshot — machine-observed field before human review
            import uot_engine_v12_patched as _eng_ref
            _pq_at_seed = dict(_eng_ref._LAST_PRIMARY_QUESTION) if _eng_ref._LAST_PRIMARY_QUESTION else {}
            _upsert_run_artifact(conn, run_id,
                initial_field_json=world_to_json(world),
                observer_basis_json=json.dumps(
                    dict(conn.execute("SELECT observer_basis FROM runs WHERE id=?",
                                      (run_id,)).fetchone() or {}).get("observer_basis") or "{}"
                ),
                primary_question_json=json.dumps(_pq_at_seed) if _pq_at_seed else None,
            )

    except Exception as exc:
        err_msg = f"[stage={stage}] {type(exc).__name__}: {str(exc)[:400]}"
        import traceback as _tb
        print(f"[PIPELINE ERROR] run={run_id} exc={exc!r}")
        _tb.print_exc()
        with get_db() as conn:
            touch_run(conn, run_id, "failed")
            record_audit(conn, run_id, "pipeline_failed", {
                "stage": stage,
                "error": err_msg,
            })


@app.post("/api/runs/{run_id}/seed")
def seed_run(run_id: str, background_tasks: BackgroundTasks):
    """
    Start the seeding pipeline as a background task and return immediately.

    The live AI pipeline takes 60-180 seconds; running it synchronously
    would hit Railway's HTTP timeout and silently drop the request.
    Instead, this endpoint returns "seeding" immediately and the frontend
    polls GET /api/runs/{run_id} until status becomes "seeded" or "failed".
    """
    with get_db() as conn:
        run = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not run:
            raise HTTPException(404, "Run not found")
        now = datetime.now(timezone.utc).isoformat()
        touch_run(conn, run_id, "seeding")
        record_audit(conn, run_id, "seeding_started", {
            "topic":     run["topic"],
            "demo_mode": json.loads(run["settings_json"] or "{}").get("demo_mode", True),
        })

    background_tasks.add_task(_run_seed_pipeline, run_id)

    return {
        "run_id":  run_id,
        "status":  "seeding",
        "message": "Pipeline running. Poll GET /api/runs/{run_id} for completion.",
    }


@app.get("/api/runs/{run_id}/graph")
def get_graph(run_id: str):
    with get_db() as conn:
        row = _get_state_for_stage(conn, run_id)
    if not row:
        raise HTTPException(404, "No graph state found. Run /seed first.")
    world  = world_from_json(row["world_json"])
    scores = json.loads(row["scores_json"]) if row["scores_json"] else None
    return _graph_to_response(world, scores)


@app.patch("/api/runs/{run_id}/graph/nodes/{node_id}")
def edit_node(run_id: str, node_id: str, req: NodeEditRequest):
    """
    Observer Review — edit a node's field values.

    All edits tracked in node.user_adjusted and audit trail.
    Numeric fields are clamped to [0,1]. Branch edits trigger renormalization.
    """
    with get_db() as conn:
        run = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not run:
            raise HTTPException(404, "Run not found")
        row = _get_state_for_stage(conn, run_id)
        if not row:
            raise HTTPException(404, "No graph state found.")

        world = world_from_json(row["world_json"])
        if node_id not in world.event_graph.nodes:
            raise HTTPException(404, f"Node {node_id} not found.")

        node    = world.event_graph.nodes[node_id]
        changes = {}
        edits   = req.model_dump(exclude_none=True)

        float_fields = {
            "probability", "temporal_energy", "temporal_coherence",
            "temporal_entropy", "indeterminacy", "record_coherence",
            "disruption_score", "observer_sensitivity", "source_agreement",
        }

        for fname, value in edits.items():
            old_val = getattr(node, fname, None)
            if fname in float_fields:
                value = clamp(float(value))
            setattr(node, fname, value)
            node.user_adjusted[fname] = value
            changes[fname] = {"from": old_val, "to": value}

        if "probability" in changes or "branch_group" in changes:
            normalize_branch_groups(world)

        params = ModelParams()
        scores = compute_instability_score(world, params)

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO world_states VALUES (?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), run_id, "reviewed",
             world_to_json(world), json.dumps(scores),
             row["edges_json"], now)
        )
        touch_run(conn, run_id, "reviewed")
        record_audit(conn, run_id, "node_edited", {
            "node_id": node_id, "changes": changes
        })
        conn.execute(
            "INSERT INTO observer_interventions VALUES (?,?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), run_id, "node_field_edit",
             "node", node_id,
             json.dumps({k: v["from"] for k, v in changes.items()}),
             json.dumps({k: v["to"]   for k, v in changes.items()}),
             now)
        )

    return {
        "node_id": node_id,
        "changes": changes,
        "graph":   _graph_to_response(world, scores),
    }


@app.put("/api/runs/{run_id}/graph/edges")
def set_edges(run_id: str, edges: List[EdgeEditRequest]):
    """
    Replace the current edge set.

    Fix 7a: after replacement, recomputes causal_support_graph,
            temporal_coherence, and temporal_energy from the new topology.
    Fix 7b: saves updated edges_json, not a stale copy from the previous row.
    """
    with get_db() as conn:
        run = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not run:
            raise HTTPException(404, "Run not found")
        row = _get_state_for_stage(conn, run_id)
        if not row:
            raise HTTPException(404, "No graph state found.")

        world     = world_from_json(row["world_json"])
        old_count = len(world.event_graph.edges)

        world.event_graph.edges = [
            EventEdge(
                source_id=e.source_id,
                target_id=e.target_id,
                relation_type=e.relation_type,
                causal_weight=clamp(e.causal_weight),
                uncertainty=clamp(e.uncertainty),
                feedback_strength=clamp(e.feedback_strength),
            )
            for e in edges
            if e.source_id in world.event_graph.nodes
            and e.target_id in world.event_graph.nodes
        ]

        # Fix 7a: recompute graph-derived causal support after topology change
        observer_basis = json.loads(run["observer_basis"])
        world = _recompute_after_edge_change(world, observer_basis)

        params = ModelParams()
        scores = compute_instability_score(world, params)

        # Fix 7b: save the new edge specs, not the stale Stage C JSON
        new_edges_json = json.dumps([
            {
                "source_id": e.source_id, "target_id": e.target_id,
                "relation_type": e.relation_type,
                "causal_weight": e.causal_weight, "uncertainty": e.uncertainty,
                "feedback_strength": e.feedback_strength,
            }
            for e in world.event_graph.edges
        ])

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO world_states VALUES (?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), run_id, "reviewed",
             world_to_json(world), json.dumps(scores),
             new_edges_json, now)
        )
        touch_run(conn, run_id, "reviewed")
        record_audit(conn, run_id, "edges_replaced", {
            "old_count": old_count,
            "new_count": len(world.event_graph.edges),
        })

    return {
        "edge_count": len(world.event_graph.edges),
        "graph":      _graph_to_response(world, scores),
    }


@app.post("/api/runs/{run_id}/evidence")
def add_evidence(run_id: str, req: AddEvidenceRequest):
    """
    Inject a Bayesian evidence event that updates a node's probability.

    In UOT: the observer injects a specific observation into the possibility
    field. The evidence collapses indeterminacy at the target node by the
    given likelihood ratio, modulated by the node's own collapse resistance.
    """
    with get_db() as conn:
        row = _get_state_for_stage(conn, run_id)
        if not row:
            raise HTTPException(404, "No graph state found.")

        world = world_from_json(row["world_json"])
        if req.target_node_id not in world.event_graph.nodes:
            raise HTTPException(404, f"Node {req.target_node_id} not found.")

        ev = EvidenceEvent(
            id=str(uuid.uuid4()),
            target_node_id=req.target_node_id,
            likelihood_ratio=req.likelihood_ratio,
            confidence=req.confidence,
            timestamp=0.0,
            source=req.source,
            description=req.description,
            applied=False,
        )
        world.evidence_events.append(ev)
        world = apply_pending_evidence(world)
        normalize_branch_groups(world)

        params = ModelParams()
        scores = compute_instability_score(world, params)

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO world_states VALUES (?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), run_id, "reviewed",
             world_to_json(world), json.dumps(scores),
             row["edges_json"], now)
        )
        touch_run(conn, run_id, "reviewed")
        record_audit(conn, run_id, "evidence_added", {
            "target_node_id":  req.target_node_id,
            "likelihood_ratio": req.likelihood_ratio,
            "confidence":       req.confidence,
            "source":           req.source,
        })
        conn.execute(
            "INSERT INTO observer_interventions VALUES (?,?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), run_id, "evidence_observation",
             "node", req.target_node_id,
             json.dumps({"likelihood_ratio": req.likelihood_ratio,
                         "confidence": req.confidence, "source": req.source}),
             json.dumps({"node_id": req.target_node_id}),
             now)
        )

    return {"evidence_id": ev.id, "graph": _graph_to_response(world, scores)}


# ══════════════════════════════════════════════════════════════════════════════
# Simulation
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/runs/{run_id}/simulate")
def run_simulation(run_id: str, req: SimulateRequest = SimulateRequest()):
    """
    Run the UOT temporal simulation.

    Fix 8: restart_from_reviewed=True (default) loads the latest reviewed/seeded
    state — a fresh run from the observer-confirmed graph.
    restart_from_reviewed=False continues from the latest simulated state.

    Fix 9: step_history is persisted inside scores_json under 'step_history' key.
    """
    with get_db() as conn:
        run = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not run:
            raise HTTPException(404, "Run not found.")

        # Load the appropriate starting state
        if req.restart_from_reviewed:
            # Use latest reviewed or seeded state (not a previous simulation)
            row = (
                _get_state_for_stage(conn, run_id, "reviewed") or
                _get_state_for_stage(conn, run_id, "seeded")
            )
        else:
            # Continue from latest state (could be a previous simulation)
            row = _get_state_for_stage(conn, run_id)

        if not row:
            raise HTTPException(400, "Graph not seeded yet. Call /seed first.")

    world  = world_from_json(row["world_json"])
    params = ModelParams()

    observer_basis    = json.loads((dict(run).get("observer_basis") or "{}")) if run else {}
    step_history      = []
    iteration_history = []

    # Save reviewed field snapshot (observer-reviewed, pre-simulation)
    with get_db() as conn:
        _upsert_run_artifact(conn, run_id,
            reviewed_field_json=world_to_json(world))

    fallback_used   = False
    loop_error_msg  = None

    if req.use_iterative_loop:
        # ── Phase 3: Iterative Discrepancy Loop ───────────────────────────────
        try:
            world, final_scores, iter_results = run_iterative_simulation(
                world, params, observer_basis,
                dt=req.dt, max_iterations=req.max_iterations
            )
        except Exception as loop_err:
            import traceback
            err_msg = f"{type(loop_err).__name__}: {loop_err}"
            print(f"[iterative loop error] {err_msg}")
            traceback.print_exc()
            if STRICT_LOOP:
                raise HTTPException(500, f"Iterative loop failed (STRICT_LOOP=true): {err_msg}")
            # Graceful fallback — returns usable result, clearly labeled
            fallback_used = True
            loop_error_msg = err_msg
            world, final_scores = world, {}
            for _ in range(min(req.steps, 3)):
                try:
                    world, final_scores = simulation_step(world, req.dt, params)
                except Exception:
                    break
            iter_results = []

        for r in iter_results:
            step_history.append({
                "step":                 r.iteration,
                "global_instability":   r.scores_after.get("global_instability", 0),
                "branch_potential":     r.scores_after.get("global_branch_potential", 0),
                "temporal_flux":        r.scores_after.get("temporal_flux", 0),
                "discrepancies_found":  len(r.discrepancies),
                "corrections_applied":  len(r.corrections),
                "converged":            r.converged,
            })
            iteration_history.append({
                "iteration":          r.iteration,
                "scores_before":      r.scores_before,
                "scores_after":       r.scores_after,
                "discrepancies":      r.discrepancies,
                "corrections":        r.corrections,
                "convergence_delta":  r.convergence_delta,
                "converged":          r.converged,
            })
    else:
        # ── Legacy fixed-step simulation ──────────────────────────────────────
        for _ in range(req.steps):
            world, scores = simulation_step(world, dt=req.dt, params=params)
            step_history.append({
                "step":               world.step,
                "global_instability": scores["global_instability"],
                "branch_potential":   scores.get("global_branch_potential", 0),
                "temporal_flux":      scores.get("temporal_flux", 0),
            })

    # ── Compute final scores and branch details ────────────────────────────────
    try:
        final_scores = compute_instability_score(world, params)
    except Exception:
        final_scores = {"global_instability": 0.3, "field_stability": "STABLE",
                        "branch_details": {}, "node_scores": {}}

    bd = final_scores.get("branch_details", {})

    # Compute convergence summary BEFORE the DB block (used in audit record)
    last_iter  = iteration_history[-1] if iteration_history else {}
    converged  = last_iter.get("converged", True) if last_iter else True
    disc_count = sum(len(r.get("discrepancies", [])) for r in iteration_history)
    corr_count = sum(len(r.get("corrections",   [])) for r in iteration_history)

    # Persist simulated world state
    step_history_json = json.dumps(step_history)
    scores_json       = json.dumps({
        **final_scores,
        "step_history":      step_history,
        "iteration_history": iteration_history,
    }, default=str)

    resolution_state = compute_resolution_state(final_scores, converged,
        iteration_history[-1].get("discrepancies", []) if iteration_history else [])

    with get_db() as conn:
        edge_row = conn.execute(
            "SELECT edges_json FROM world_states WHERE run_id=? ORDER BY created_at DESC LIMIT 1",
            (run_id,)
        ).fetchone()
        edges_json = edge_row["edges_json"] if edge_row else "[]"
        now = datetime.now(timezone.utc).isoformat()

        # Save simulated world state
        conn.execute(
            "INSERT INTO world_states VALUES (?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), run_id, "simulated",
             world_to_json(world), scores_json, edges_json, now)
        )
        touch_run(conn, run_id, "simulated")

        # Save iteration logs — each simulate call = one attempt_index
        prior = conn.execute(
            "SELECT MAX(attempt_index) FROM iteration_logs WHERE run_id=?", (run_id,)
        ).fetchone()
        attempt_idx = (prior[0] or 0) + 1

        for r in iteration_history:
            conn.execute(
                """INSERT INTO iteration_logs
                   (id, run_id, attempt_index, iteration_index,
                    scores_before_json, scores_after_json,
                    discrepancies_json, near_discrepancies_json,
                    corrections_json, convergence_delta_json,
                    fallback_used, loop_error, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (str(uuid.uuid4()), run_id, attempt_idx, r["iteration"],
                 json.dumps(r.get("scores_before", {})),
                 json.dumps(r.get("scores_after",  {})),
                 json.dumps(r.get("discrepancies",      [])),
                 json.dumps(r.get("near_discrepancies", [])),
                 json.dumps(r.get("corrections",        [])),
                 json.dumps(r.get("convergence_delta",  {})),
                 1 if fallback_used else 0,
                 loop_error_msg,
                 now)
            )

        # Save run artifact (convergence summary)
        convergence_summary = {
            "iterations_run":       len(iteration_history),
            "loop_converged":       converged,
            "resolution_state":     resolution_state,
            "discrepancies_found":  disc_count,
            "corrections_applied":  corr_count,
            "fallback_used":        fallback_used,
            "loop_error":           loop_error_msg,
            "final_instability":    final_scores.get("global_instability"),
            "field_stability":      final_scores.get("field_stability"),
        }
        # Retrieve PrimaryQuestion — prefer DB artifact (set during seeding) over module var
        import uot_engine_v12_patched as _eng
        _art_row = conn.execute(
            "SELECT primary_question_json FROM run_artifacts WHERE run_id=?", (run_id,)
        ).fetchone()
        _pq_json = _art_row["primary_question_json"] if _art_row else None
        if _pq_json:
            try:
                pq = json.loads(_pq_json)
            except Exception:
                pq = {}
        else:
            pq = dict(_eng._LAST_PRIMARY_QUESTION) if _eng._LAST_PRIMARY_QUESTION else {}

        # Last-resort inference: if pq still empty, build from the simulation result's branch groups
        if not pq.get('primary_branch_group_id'):
            bg_members: dict = {}
            for node in (world.event_graph.nodes.values() if hasattr(world.event_graph.nodes, 'values') else world.event_graph.nodes):
                bg = getattr(node, 'branch_group', None)
                if bg:
                    bg_members.setdefault(bg, []).append(node)
            multi = {k: v for k, v in bg_members.items() if len(v) > 1}
            if multi:
                primary_bg = max(multi, key=lambda k: len(multi[k]))
                import re as _re3
                def _slot(s):
                    s = _re3.sub(r'[^a-z0-9 ]', '', str(s).lower())
                    return _re3.sub(r' +', '_', s.strip())[:40] or 'outcome'
                slots = [_slot(getattr(n, 'branch_label', '') or getattr(n, 'id', ''))
                         for n in sorted(multi[primary_bg], key=lambda n: -(getattr(n,'probability',0)))][:4]
                pq = {
                    'question_text': req.topic if hasattr(req, 'topic') else '',
                    'primary_branch_group_id': primary_bg,
                    'canonical_outcome_slots': slots,
                    'confidence': 0.65,
                }

        import dataclasses as _dc
        _sa_json = json.dumps([_dc.asdict(sa) for sa in (_slot_assignments or [])]) if _slot_assignments else None
        _si_json = json.dumps([_dc.asdict(si) for si in (_slot_influences  or [])]) if _slot_influences  else None
        _upsert_run_artifact(conn, run_id,
            final_field_json=world_to_json(world),
            convergence_summary_json=json.dumps(convergence_summary),
            primary_question_json=json.dumps(pq) if pq else None,
            slot_assignments_json=_sa_json,
            slot_influences_json=_si_json)

        # Phase 6.7.3: capture slot snapshots AFTER final_field_json/
        # primary_question_json are updated to the POST-SIMULATION state above.
        # Previously this ran before that update, so capture_slot_snapshots'
        # primary path (which reads final_field_json from run_artifacts) saw
        # the pre-simulation seeded probabilities — the Tracked Outcomes modal
        # showed seed-time values, not the post-simulation branch group the
        # user just computed. Also benefits from the freshly-resolved pq
        # (including the last-resort inference above) for bg_id lookup.
        run_row = conn.execute("SELECT series_id FROM runs WHERE id=?", (run_id,)).fetchone()
        if run_row and run_row["series_id"]:
            try:
                capture_slot_snapshots(conn, run_id, run_row["series_id"], world, final_scores or {})
            except Exception as _snap_err:
                print(f"[Phase 6] slot_snapshot error: {_snap_err}")

        record_audit(conn, run_id, "simulation_complete", {
            "steps":              len(step_history),
            "iterations_run":     len(iteration_history),
            "loop_converged":     converged,
            "discrepancies_found": disc_count,
            "corrections_applied": corr_count,
            "used_iterative_loop": req.use_iterative_loop,
            "global_instability": final_scores.get("global_instability"),
            "field_stability":    final_scores.get("field_stability"),
        })

    # ── Build and return response ──────────────────────────────────────────────

    return {
        "run_id":              run_id,
        "primary_question":    pq if pq else None,
        "steps_run":           len(step_history),
        "iterations_run":      len(iteration_history),
        "forward_steps_per_iteration": 1,
        "total_forward_steps": len(iteration_history),
        "loop_converged":      converged,
        "resolution_state":    resolution_state,
        "simulation_mode":     "fallback_simple" if fallback_used else "iterative",
        "fallback_used":       fallback_used,
        "loop_error":          loop_error_msg,
        "discrepancies_found": disc_count,
        "corrections_applied": corr_count,
        "step_history":        step_history,
        "iteration_history":   iteration_history,
        "graph":               _graph_to_response(world),
        "field_stability":     final_scores.get("field_stability", "STABLE"),
        "global_instability":  final_scores.get("global_instability", 0.0),
        "branch_details":      bd,
        "restarted":           req.restart_from_reviewed,
        "topic":               dict(run)["topic"] if run else "",
        "settings":            json.loads(run["settings_json"] or "{}") if run else {},
        "primary_question":    {
            "primary_branch_group_id": pq.get("primary_branch_group_id", "primary_outcome"),
            "seed_quality": pq.get("seed_quality"),
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# Root — serve frontend HTML
# ═══════════════════════════════════════════════════════════════════════
@app.get("/")
def root():
    """Serve the single-page frontend."""
    html_path = Path(__file__).parent / "uot_frontend_v2.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>UOT/TEE</h1><p>Frontend not found.</p>", status_code=404)


# ═══════════════════════════════════════════════════════════════════════
# Run status polling
# ═══════════════════════════════════════════════════════════════════════
@app.get("/api/runs/{run_id}/status")
def get_run_status(run_id: str):
    """Frontend polls this during seeding. Returns status + last error."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT status, last_error FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
    if not row:
        raise HTTPException(404, "Run not found")
    err = row["last_error"]
    try:
        err_obj = json.loads(err) if err else None
    except Exception:
        err_obj = {"error": err} if err else None
    return {"run_id": run_id, "status": row["status"], "last_error": err_obj}


# ═══════════════════════════════════════════════════════════════════════
# Audit trail
# ═══════════════════════════════════════════════════════════════════════
@app.get("/api/runs/{run_id}/audit")
def get_audit_trail(run_id: str):
    """Return the full audit log for a run."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM audit_events WHERE run_id = ? ORDER BY timestamp ASC",
            (run_id,)
        ).fetchall()
    return {"run_id": run_id, "entries": [dict(r) for r in rows]}


# ═══════════════════════════════════════════════════════════════════════
# Restart from reviewed state
# ═══════════════════════════════════════════════════════════════════════
@app.post("/api/runs/{run_id}/restart_from_reviewed")
def restart_from_reviewed(run_id: str):
    """Reset simulation state to the last reviewed world snapshot."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT world_json FROM world_states WHERE run_id = ? AND stage = 'reviewed' "
            "ORDER BY created_at DESC LIMIT 1",
            (run_id,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "No reviewed state found for this run")
        conn.execute(
            "DELETE FROM world_states WHERE run_id = ? AND stage = 'simulated'",
            (run_id,)
        )
        touch_run(conn, run_id, "reviewed")
    return {"run_id": run_id, "status": "restored_to_reviewed"}


# ═══════════════════════════════════════════════════════════════════════
# Basis suggestion
# ═══════════════════════════════════════════════════════════════════════
@app.post("/api/basis-suggest", response_model=BasisSuggestResponse)
def suggest_basis(req: BasisSuggestRequest):
    """
    Pre-seeding AI calibration: suggest 4-6 observer dimensions for a topic.
    Called from the landing page before run creation so the user can review
    their measurement basis before the source field is collected.
    """
    hl = req.horizon_label
    hc = req.horizon_custom
    if hl == "custom" and hc:
        horizon_desc = hc
        target_date  = "custom timeframe"
    else:
        h = HorizonConfig.from_label(hl)
        horizon_desc = h.horizon_label
        target_date  = h.target_date_str

    prompt = (
        f"Given this scenario topic and time horizon, suggest 4-6 observer measurement "
        f"basis dimensions.\n\nThe measurement basis is the observer's interpretive frame: "
        f"the dimensions along which the engine notices salience, uncertainty, causality, "
        f"and temporal pressure.\n\nTopic: {req.topic}\nHorizon: {horizon_desc}, "
        f"ending {target_date}\n\nReturn JSON only:\n"
        '{"basis": [{"key": "snake_case_key", "label": "Human readable label", '
        '"description": "One sentence description.", "weight": 0.75}]}'
        "\n\nReturn 4-6 dimensions. Keys must be unique snake_case. Weights 0.3-0.95."
    )
    try:
        result = call_anthropic_api(
            "You are a UOT observer calibration assistant. Return only valid JSON.",
            prompt, model="claude-sonnet-4-6", max_tokens=1500, timeout=30
        )
        if not isinstance(result, dict) or "basis" not in result:
            raise ValueError("Unexpected response")
        basis = [
            BasisDimension(
                key=str(b.get("key", f"dim_{i}")).lower().replace(" ", "_"),
                label=str(b.get("label", f"Dimension {i+1}")),
                description=str(b.get("description", "")),
                weight=float(b.get("weight", 0.7)) if isinstance(b.get("weight"), (int, float)) else 0.7,
            )
            for i, b in enumerate(result.get("basis", []))
            if isinstance(b, dict)
        ][:6]
        if not basis:
            raise ValueError("No valid dimensions")
        return BasisSuggestResponse(topic=req.topic, basis=basis)
    except Exception as e:
        import traceback
        print(f"[basis-suggest error] {type(e).__name__}: {e}")
        traceback.print_exc()
        return BasisSuggestResponse(topic=req.topic, basis=[
            BasisDimension(key="primary_outcome",         label="Primary outcome likelihood",  description="How likely the main scenario resolution is.", weight=0.8),
            BasisDimension(key="institutional_stability", label="Institutional stability",     description="Whether key institutions hold or fracture.",   weight=0.7),
            BasisDimension(key="escalation_risk",         label="Escalation risk",             description="Probability of events intensifying.",          weight=0.65),
            BasisDimension(key="diplomatic_pressure",     label="Diplomatic pressure",         description="Strength of negotiation and resolution forces.", weight=0.6),
        ])


# ═══════════════════════════════════════════════════════════════════════
# Iteration logs diagnostic endpoint
# ═══════════════════════════════════════════════════════════════════════
@app.get("/api/runs/{run_id}/iteration-logs")
def get_iteration_logs(run_id: str):
    """
    Return full iteration log for a run — every discrepancy, near-discrepancy,
    correction, and convergence delta from each loop pass.
    Useful for threshold calibration and loop debugging.
    """
    with get_db() as conn:
        run = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not run:
            raise HTTPException(404, "Run not found")
        logs = conn.execute(
            "SELECT * FROM iteration_logs WHERE run_id=? ORDER BY iteration_index ASC",
            (run_id,)
        ).fetchall()
        artifact = conn.execute(
            "SELECT convergence_summary_json FROM run_artifacts WHERE run_id=? LIMIT 1",
            (run_id,)
        ).fetchone()

    def _parse(row):
        return {
            "iteration_index":        row["iteration_index"],
            "scores_before":          json.loads(row["scores_before_json"]   or "{}"),
            "scores_after":           json.loads(row["scores_after_json"]    or "{}"),
            "discrepancies":          json.loads(row["discrepancies_json"]   or "[]"),
            "near_discrepancies":     json.loads(row["near_discrepancies_json"] or "[]"),
            "corrections":            json.loads(row["corrections_json"]     or "[]"),
            "convergence_delta":      json.loads(row["convergence_delta_json"] or "{}"),
            "fallback_used":          bool(row["fallback_used"]),
            "loop_error":             row["loop_error"],
        }

    conv_summary = json.loads(artifact["convergence_summary_json"] or "{}") if artifact else {}
    return {
        "run_id":             run_id,
        "convergence_summary": conv_summary,
        "iteration_count":    len(logs),
        "iterations":         [_parse(r) for r in logs],
    }


# ═══════════════════════════════════════════════════════════════════════
# Research artifact export endpoint
# ═══════════════════════════════════════════════════════════════════════
@app.get("/api/runs/{run_id}/artifact")
def get_run_artifact(run_id: str):
    """
    Return the complete research artifact for a run:
    initial field → reviewed field → final field → convergence summary.
    This is the spine of UOT replayability.
    """
    with get_db() as conn:
        run = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not run:
            raise HTTPException(404, "Run not found")
        artifact = conn.execute(
            "SELECT * FROM run_artifacts WHERE run_id=? LIMIT 1", (run_id,)
        ).fetchone()
        interventions = conn.execute(
            "SELECT * FROM observer_interventions WHERE run_id=? ORDER BY timestamp ASC",
            (run_id,)
        ).fetchall()

    if not artifact:
        raise HTTPException(404, "No artifact found for this run. Run a simulation first.")

    return {
        "run_id":               run_id,
        "topic":                dict(run).get("topic"),
        "created_at":           dict(run).get("created_at"),
        "observer_basis":       json.loads(artifact["observer_basis_json"] or "{}"),
        "initial_field":        artifact["initial_field_json"],
        "reviewed_field":       artifact["reviewed_field_json"],
        "final_field":          artifact["final_field_json"],
        "convergence_summary":  json.loads(artifact["convergence_summary_json"] or "{}"),
        "observer_interventions": [dict(i) for i in interventions],
    }


# ═══════════════════════════════════════════════════════════════════════
# Research Diagnostics / Calibration Lab
# ═══════════════════════════════════════════════════════════════════════
@app.get("/api/diagnostics/summary")
def diagnostics_summary():
    """
    Aggregate statistics across all completed runs.
    Implements the Calibration Lab scaffold from Phase 3.6.
    For research/developer use — not part of main user flow.
    """
    with get_db() as conn:
        runs = conn.execute(
            "SELECT id, status, created_at FROM runs ORDER BY created_at DESC"
        ).fetchall()
        artifacts = conn.execute(
            """SELECT ra.convergence_summary_json
               FROM run_artifacts ra
               INNER JOIN runs r ON ra.run_id = r.id
               WHERE ra.convergence_summary_json IS NOT NULL"""
        ).fetchall()
        # Phase 6.5 (Pass 5): seed quality / organic rate history
        seed_quality_rows = conn.execute(
            """SELECT ra.primary_question_json, r.created_at
               FROM run_artifacts ra
               INNER JOIN runs r ON ra.run_id = r.id
               WHERE ra.primary_question_json IS NOT NULL
               ORDER BY r.created_at DESC LIMIT 50"""
        ).fetchall()
        iter_logs = conn.execute(
            """SELECT il.discrepancies_json, il.near_discrepancies_json,
                      il.scores_after_json, il.fallback_used
               FROM iteration_logs il
               INNER JOIN runs r ON il.run_id = r.id"""
        ).fetchall()
        interventions = conn.execute(
            """SELECT oi.intervention_type, COUNT(*) as cnt
               FROM observer_interventions oi
               INNER JOIN runs r ON oi.run_id = r.id
               GROUP BY oi.intervention_type"""
        ).fetchall()

    total_runs = len(runs)
    completed   = sum(1 for r in runs if dict(r).get("status") == "simulated")

    # Parse convergence summaries
    summaries = []
    for a in artifacts:
        try:
            s = json.loads(a["convergence_summary_json"])
            if s: summaries.append(s)
        except Exception:
            pass

    resolution_counts = {}
    iter_counts, instabilities, disc_counts = [], [], []
    fallback_count = 0
    for s in summaries:
        rs = s.get("resolution_state", "unknown")
        resolution_counts[rs] = resolution_counts.get(rs, 0) + 1
        if s.get("iterations_run") is not None:
            iter_counts.append(s["iterations_run"])
        if s.get("final_instability") is not None:
            instabilities.append(s["final_instability"])
        disc_counts.append(s.get("discrepancies_found", 0))
        if s.get("fallback_used"):
            fallback_count += 1

    # Discrepancy type breakdown from iteration logs
    disc_by_type = {}
    near_by_type = {}
    corrected_nodes = {}
    for row in iter_logs:
        if row["fallback_used"]:
            continue
        try:
            discs = json.loads(row["discrepancies_json"] or "[]")
            for d in discs:
                t = d.get("type", "unknown")
                disc_by_type[t] = disc_by_type.get(t, 0) + 1
                for nid in d.get("node_ids", []):
                    corrected_nodes[nid] = corrected_nodes.get(nid, 0) + 1
        except Exception: pass
        try:
            nears = json.loads(row["near_discrepancies_json"] or "[]")
            for d in nears:
                t = d.get("type", "unknown")
                near_by_type[t] = near_by_type.get(t, 0) + 1
        except Exception: pass

    def avg(lst): return round(sum(lst) / len(lst), 4) if lst else None

    # Phase 6.5 (Pass 5): organic rate / seed quality trend
    seed_quality_history = []
    organic_rates = []
    quality_flag_counts = {"green": 0, "amber": 0, "red": 0}
    synthetic_reason_totals = {}
    for row in seed_quality_rows:
        try:
            pq = json.loads(row["primary_question_json"] or "{}")
            sq = pq.get("seed_quality")
            if not sq: continue
            seed_quality_history.append({
                "created_at": row["created_at"],
                "organic_rate": sq.get("organic_rate"),
                "synthetic_slots": sq.get("synthetic_slots"),
                "unmapped_slots": sq.get("unmapped_slots"),
                "quality_flag": sq.get("quality_flag"),
            })
            if sq.get("organic_rate") is not None:
                organic_rates.append(sq["organic_rate"])
            qf = sq.get("quality_flag")
            if qf in quality_flag_counts:
                quality_flag_counts[qf] += 1
        except Exception:
            pass

    return {
        "total_runs":               total_runs,
        "completed_simulations":    completed,
        "resolution_state_counts":  resolution_counts,
        "avg_iterations_run":       avg(iter_counts),
        "avg_global_instability":   avg(instabilities),
        "avg_discrepancies_per_run": avg(disc_counts),
        "fallback_count":           fallback_count,
        "discrepancies_by_type":    disc_by_type,
        "near_discrepancies_by_type": near_by_type,
        "observer_interventions_by_type": {dict(r)["intervention_type"]: dict(r)["cnt"] for r in interventions},
        "thresholds_note": (
            f"Calibration requires 15-25 clean runs. Currently {completed} completed. "
            "Do not adjust thresholds until minimum reached."
        ),
        "seed_quality": {
            "avg_organic_rate": avg(organic_rates),
            "quality_flag_counts": quality_flag_counts,
            "recent_history": seed_quality_history[:20],
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# Health check
# ═══════════════════════════════════════════════════════════════════════
@app.post("/api/admin/clear-preloop-runs")
def clear_preloop_runs():
    """
    Delete all runs that have no iteration_logs entries — these are pre-loop runs
    that predate the iterative discrepancy loop implementation.
    Research runs with iteration_logs are preserved.
    """
    with get_db() as conn:
        # Find run_ids that have NO iteration_logs entries
        pre_loop = conn.execute("""
            SELECT r.id FROM runs r
            WHERE NOT EXISTS (
                SELECT 1 FROM iteration_logs il WHERE il.run_id = r.id
            )
        """).fetchall()
        deleted_ids = [row["id"] for row in pre_loop]
        for rid in deleted_ids:
            conn.execute("DELETE FROM iteration_logs   WHERE run_id = ?", (rid,))
            conn.execute("DELETE FROM run_artifacts    WHERE run_id = ?", (rid,))
            conn.execute("DELETE FROM observer_interventions WHERE run_id = ?", (rid,))
            conn.execute("DELETE FROM runs WHERE id = ?", (rid,))
    return {
        "deleted_count": len(deleted_ids),
        "deleted_run_ids": deleted_ids[:10],   # show first 10 for verification
        "message": f"Cleared {len(deleted_ids)} pre-loop runs. Runs with iteration logs preserved.",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Phase 6: Scenario Series — Longitudinal Tracking Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/series")
def create_series(req: dict):
    """
    Create a new ScenarioSeries from a completed run.
    Called when the user clicks "Track this scenario."
    """
    run_id  = req.get("run_id", "")
    title   = req.get("title", "")
    policy  = req.get("refresh_policy", "manual")
    if not run_id:
        raise HTTPException(400, "run_id required")

    with get_db() as conn:
        run = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not run:
            raise HTTPException(404, "Run not found")

        art = conn.execute(
            "SELECT * FROM run_artifacts WHERE run_id=?", (run_id,)
        ).fetchone()

        series_id = "SRS_" + str(uuid.uuid4()).replace("-","")[:12].upper()
        now = datetime.now(timezone.utc).isoformat()

        run_settings = dict(run).get("settings_json", "{}")
        conn.execute(
            """INSERT INTO scenario_series
               (series_id, title, original_question, normalized_question,
                horizon_config_json, observer_basis_json, primary_question_json,
                canonical_slots_json, created_at, last_updated_at, status, refresh_policy)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (series_id,
             title or dict(run).get("topic", "Untitled scenario"),
             dict(run).get("topic", ""),
             dict(run).get("topic", "").lower().strip(),
             run_settings,           # stores horizon + horizon_custom
             dict(run).get("observer_basis", "{}"),
             art["primary_question_json"] if art else None,
             _extract_canonical_slots_json(art),
             now, now, "active", policy)
        )

        # Link the originating run to this series
        run_count = conn.execute(
            "SELECT COUNT(*) FROM runs WHERE series_id=?", (series_id,)
        ).fetchone()[0]
        conn.execute(
            "UPDATE runs SET series_id=?, run_index=?, trigger_type=? WHERE id=?",
            (series_id, run_count + 1, "manual", run_id)
        )
        conn.commit()

        # Capture slot snapshots for the originating run
        try:
            worlds = conn.execute(
                "SELECT world_json, scores_json FROM world_states WHERE run_id=? AND stage='simulated' LIMIT 1",
                (run_id,)
            ).fetchone()
            if worlds:
                world = world_from_json(worlds["world_json"])
                sc = json.loads(worlds["scores_json"] or "{}")
                capture_slot_snapshots(conn, run_id, series_id, world, sc)
        except Exception as e:
            print(f"[Phase 6] Initial snapshot error: {e}")
        conn.commit()

    return {"series_id": series_id, "run_id": run_id, "status": "created"}


@app.get("/api/series")
def list_series():
    """List all tracked scenario series."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM scenario_series ORDER BY last_updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/series/{series_id}/timeline")
def get_series_timeline(series_id: str):
    """
    Return the slot probability timeline for a scenario series.
    Used to render the longitudinal probability chart.
    """
    with get_db() as conn:
        series = conn.execute(
            "SELECT * FROM scenario_series WHERE series_id=?", (series_id,)
        ).fetchone()
        if not series:
            raise HTTPException(404, "Series not found")

        runs = conn.execute(
            """SELECT r.id, r.run_index, r.created_at, r.trigger_type,
                      a.convergence_summary_json
               FROM runs r
               LEFT JOIN run_artifacts a ON a.run_id = r.id
               WHERE r.series_id=?
               ORDER BY r.run_index""",
            (series_id,)
        ).fetchall()

        snapshots = conn.execute(
            """SELECT ss.*, r.run_index, r.created_at as run_date
               FROM slot_snapshots ss
               JOIN runs r ON r.id = ss.run_id
               WHERE ss.series_id=?
               ORDER BY r.run_index, ss.slot_id""",
            (series_id,)
        ).fetchall()

        deltas = conn.execute(
            "SELECT * FROM field_deltas WHERE series_id=? ORDER BY created_at",
            (series_id,)
        ).fetchall()

    # Phase 7.4: attach resonance data per run, ordered for timeline display
    _resonance_by_run = {}
    for _r in runs:
        try:
            _res_json = dict(_r).get("series_resonance_json")  # safe: .get() never throws KeyError
        except Exception:
            _res_json = None
        if _res_json:
            try:
                _resonance_by_run[_r["id"]] = json.loads(_res_json)
            except Exception:
                pass

    return {
        "series":    dict(series),
        "runs":      [dict(r) for r in runs],
        "snapshots": [dict(s) for s in snapshots],
        "deltas":    [dict(d) for d in deltas],
        "resonance_by_run": _resonance_by_run,
    }



@app.post("/api/runs/{run_id}/link-series")
def link_run_to_series(run_id: str, req: dict):
    """Link a re-seeded run to an existing scenario series."""
    series_id = req.get("series_id", "")
    run_index = int(req.get("run_index", 1))
    if not series_id:
        raise HTTPException(400, "series_id required")
    with get_db() as conn:
        conn.execute(
            "UPDATE runs SET series_id=?, run_index=?, trigger_type=? WHERE id=?",
            (series_id, run_index, "manual_reseed", run_id)
        )
        conn.execute(
            "UPDATE scenario_series SET last_updated_at=? WHERE series_id=?",
            (datetime.now(timezone.utc).isoformat(), series_id)
        )
        # Capture slot snapshots for this new run
        worlds = conn.execute(
            "SELECT world_json, scores_json FROM world_states WHERE run_id=? AND stage='simulated' LIMIT 1",
            (run_id,)
        ).fetchone()
        if worlds:
            try:
                # Only capture if not already done by the simulation hook
                existing_snaps = conn.execute(
                    "SELECT COUNT(*) FROM slot_snapshots WHERE run_id=? AND series_id=?",
                    (run_id, series_id)
                ).fetchone()[0]
                if existing_snaps == 0:
                    world = world_from_json(worlds["world_json"])
                    sc = json.loads(worlds["scores_json"] or "{}")
                    capture_slot_snapshots(conn, run_id, series_id, world, sc)
                else:
                    print(f"[Phase 6] Snapshots already captured for run {run_id}, skipping duplicate")
            except Exception as e:
                print(f"[Phase 6] Reseed snapshot error: {e}")
        conn.commit()
    return {"linked": True, "run_id": run_id, "series_id": series_id}


@app.delete("/api/series/{series_id}")
def delete_series(series_id: str):
    """Delete a tracked series (does not delete the underlying runs)."""
    with get_db() as conn:
        conn.execute("DELETE FROM slot_snapshots WHERE series_id=?", (series_id,))
        conn.execute("DELETE FROM field_deltas   WHERE series_id=?", (series_id,))
        conn.execute("UPDATE runs SET series_id=NULL, run_index=NULL WHERE series_id=?",
                     (series_id,))
        conn.execute("DELETE FROM scenario_series WHERE series_id=?", (series_id,))
        conn.commit()
    return {"deleted": series_id}



def _extract_canonical_slots_json(art) -> str:
    """
    Extract canonical slots from run_artifacts as a JSON array of full slot objects.
    Phase 6: reads canonical_slots_full first (has proper labels), falls back to slot IDs.
    """
    if not art or not art["primary_question_json"]:
        return "[]"
    try:
        pq = json.loads(art["primary_question_json"])
        # Prefer full slot objects (set by normalize_and_aggregate_primary_slots)
        full = pq.get("canonical_slots_full", [])
        if full and isinstance(full, list) and len(full) > 0:
            if isinstance(full[0], dict) and full[0].get("slot_id"):
                return json.dumps(full)
        # Fallback: bare slot ID strings — wrap in minimal objects
        slots = pq.get("canonical_outcome_slots", [])
        if not slots:
            return "[]"
        if isinstance(slots[0], str):
            return json.dumps([
                {"slot_id": s,
                 "label":   s.replace("_", " ").title(),
                 "slot_polarity": "yes",
                 "slot_kind": "outcome"}
                for s in slots
            ])
        return json.dumps(slots) if isinstance(slots, list) else "[]"
    except Exception:
        return "[]"



@app.post("/api/admin/reset-calibration")
def reset_calibration():
    """
    Delete all runs except the 7 most recent.
    Resets Calibration Lab baseline after Phase 5 architectural changes.
    """
    keep = 7
    print(f"[ADMIN] reset-calibration called, keeping last {keep} runs")
    with get_db() as conn:
        # Keep the `keep` most recent simulated runs, delete the rest
        all_runs = conn.execute(
            "SELECT id FROM runs ORDER BY created_at DESC"
        ).fetchall()
        runs_to_keep = {row["id"] for row in all_runs[:keep]}
        all_runs_ids = [r["id"] for r in all_runs]
        runs_to_delete = [rid for rid in all_runs_ids if rid not in runs_to_keep]
        for rid in runs_to_delete:
            conn.execute("DELETE FROM iteration_logs        WHERE run_id=?", (rid,))
            conn.execute("DELETE FROM run_artifacts         WHERE run_id=?", (rid,))
            conn.execute("DELETE FROM audit_events          WHERE run_id=?", (rid,))
            conn.execute("DELETE FROM observer_interventions WHERE run_id=?", (rid,))
            conn.execute("DELETE FROM world_states          WHERE run_id=?", (rid,))
            conn.execute("DELETE FROM slot_snapshots        WHERE run_id=?", (rid,))
            conn.execute("DELETE FROM field_deltas          WHERE from_run_id=? OR to_run_id=?", (rid, rid))
            conn.execute("DELETE FROM runs                  WHERE id=?",     (rid,))
        conn.commit()
    return {"deleted": len(runs_to_delete), "kept": len(runs_to_keep),
            "message": f"Calibration reset: {len(runs_to_delete)} old runs removed, {len(runs_to_keep)} recent runs kept."}

def capture_slot_snapshots(conn, run_id: str, series_id: str, world=None, scores: dict = None):
    """
    Capture per-slot probabilities for each primary branch group member.
    Primary path: reads from final_field_json in run_artifacts (event dicts, reliable).
    Fallback: uses the world object (WorldTimelineState.event_graph.nodes).
    """
    import uuid as _uuid
    now = datetime.now(timezone.utc).isoformat()
    scores = {} if scores is None else scores  # single assignment avoids Python 3.12 scoping issue

    # Get primary branch group ID from run_artifacts
    art = conn.execute(
        "SELECT final_field_json, primary_question_json FROM run_artifacts WHERE run_id=?",
        (run_id,)
    ).fetchone()
    bg_id = ""
    event_dicts = []

    if art:
        if art["primary_question_json"]:
            try:
                pq = json.loads(art["primary_question_json"])
                bg_id = pq.get("primary_branch_group_id", "")
            except Exception:
                pass
        if art["final_field_json"]:
            try:
                raw = json.loads(art["final_field_json"])
                # WorldTimelineState serializes as: {event_graph: {nodes: {...}, edges: [...]}, ...}
                eg = raw.get("event_graph", raw) if isinstance(raw, dict) else {}
                nodes_raw = eg.get("nodes", {}) if isinstance(eg, dict) else {}
                if isinstance(nodes_raw, dict):
                    event_dicts = list(nodes_raw.values())
            except Exception as e:
                print(f"[Phase 6] final_field_json parse error: {e}")

    # Fallback: use world object (WorldTimelineState.event_graph.nodes)
    if not event_dicts and world is not None:
        try:
            eg = getattr(world, "event_graph", None)
            if eg is None:
                eg = world   # in case caller passed graph directly
            raw_nodes = getattr(eg, "nodes", {})
            if isinstance(raw_nodes, dict):
                event_dicts = []
                for n in raw_nodes.values():
                    event_dicts.append({
                        "id":             getattr(n, "id", ""),
                        "label":          getattr(n, "label", ""),
                        "branch_group":   getattr(n, "branch_group", None),
                        "branch_label":   getattr(n, "branch_label", None),
                        "probability":    getattr(n, "probability", 0.5),
                        "branch_probability": getattr(n, "branch_probability", None),
                        "canonical_slot_id":  getattr(n, "canonical_slot_id", None),
                        "temporal_status":getattr(n, "temporal_status", "unresolved"),
                        "outcome_role":   getattr(n, "outcome_role", "causal_context"),
                    })
        except Exception as e:
            print(f"[Phase 6] world object fallback error: {e}")

    # Filter to primary branch group members
    bg_events = [e for e in event_dicts
                 if isinstance(e, dict) and e.get("branch_group") == bg_id] if bg_id else []
    if not bg_events and bg_id:
        # bg_id was set but no events matched — nothing to capture
        print(f"[Phase 6] Primary branch group '{bg_id}' has no events for run {run_id}")
    elif not bg_events:
        # No bg_id: find the branch group with the most members (likely the primary)
        from collections import Counter
        bg_counts = Counter(e.get("branch_group") for e in event_dicts
                            if isinstance(e, dict) and e.get("branch_group"))
        if bg_counts:
            largest_bg = bg_counts.most_common(1)[0][0]
            bg_events = [e for e in event_dicts
                         if isinstance(e, dict) and e.get("branch_group") == largest_bg]

    if not bg_events:
        print(f"[Phase 6] No branch group events for run {run_id} "
              f"(bg_id={bg_id!r}, event_dicts={len(event_dicts)})")
        return

    captured = 0
    for ev in bg_events:
        nid       = str(ev.get("id", ""))
        # Use canonical_slot_id if present (set by normalize_and_aggregate_primary_slots)
        # Fall back to branch_label for backwards compatibility
        slot_id   = str(ev.get("canonical_slot_id") or ev.get("branch_label") or ev.get("label") or nid)[:120]
        # branch_label = slot.label (canonical, consistent across observations)
        # label = original organic event text (varies run to run)
        slot_label = str(ev.get("branch_label") or ev.get("label") or slot_id)[:120]
        # Phase 6.6 (GPT guidance): tracked outcomes represent the slot's NORMALIZED
        # share of the primary branch group (branch_probability), not the
        # representative event's raw independent probability. branch_probability
        # is computed by compute_branch_probabilities() during simulation and
        # falls back to the raw probability for older runs that predate this field.
        _bp = ev.get("branch_probability")
        prob = float(_bp) if _bp is not None else float(ev.get("probability", 0.5) or 0.5)
        node_sc   = scores.get(nid, {})
        conn.execute(
            """INSERT OR REPLACE INTO slot_snapshots
               (id,series_id,run_id,slot_id,slot_label,slot_polarity,slot_kind,
                probability,temporal_energy,temporal_coherence,temporal_entropy,
                indeterminacy,assigned_event_id,event_label,synthetic,
                resolved_status,record_confidence,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (str(_uuid.uuid4()), series_id, run_id,
             slot_id, slot_label,
             str(ev.get("outcome_role", "primary_outcome")),
             "outcome", prob,
             float(node_sc.get("temporal_energy",   0) or 0),
             float(node_sc.get("temporal_coherence", 0) or 0),
             float(node_sc.get("temporal_entropy",   0) or 0),
             float(node_sc.get("indeterminacy",      0) or 0),
             nid, slot_label,
             1 if nid.startswith("EVT_SYNTH") else 0,
             str(ev.get("temporal_status", "unresolved")),
             float(node_sc.get("record_coherence", 0.5) or 0.5),
             now)
        )
        captured += 1

    # Phase 6.6.2: carry forward snapshots for canonical slots that were
    # UNMAPPED this observation (Stage C found no organic match and
    # synthetic_allowed=False, so no event/snapshot exists for this slot_id
    # this round). Without this, field_deltas has no row for the unmapped
    # slot this observation, and the "Recent Field Deltas" display (a
    # fixed-size window of recent rows) backfills with a stale row from an
    # earlier observation pair — producing duplicate/missing entries.
    # Carrying forward the previous branch_probability represents "no new
    # evidence this round" with delta=0.0, which is UOT-consistent.
    covered_slot_ids = set()
    for ev in bg_events:
        _sid = str(ev.get("canonical_slot_id") or ev.get("branch_label") or ev.get("label") or ev.get("id"))[:120]
        covered_slot_ids.add(_sid)

    all_slot_ids = []
    if art and art["primary_question_json"]:
        try:
            _pq = json.loads(art["primary_question_json"])
            all_slot_ids = _pq.get("canonical_outcome_slots", []) or []
        except Exception:
            pass

    for _sid in all_slot_ids:
        if _sid in covered_slot_ids:
            continue
        prior_row = conn.execute(
            """SELECT slot_label, slot_polarity, slot_kind, probability, temporal_energy,
                      temporal_coherence, temporal_entropy, indeterminacy, assigned_event_id,
                      event_label, synthetic, resolved_status, record_confidence
               FROM slot_snapshots
               WHERE series_id=? AND slot_id=? AND run_id!=?
               ORDER BY created_at DESC LIMIT 1""",
            (series_id, _sid, run_id)
        ).fetchone()
        if not prior_row:
            continue  # never captured before — nothing to carry forward
        conn.execute(
            """INSERT OR REPLACE INTO slot_snapshots
               (id,series_id,run_id,slot_id,slot_label,slot_polarity,slot_kind,
                probability,temporal_energy,temporal_coherence,temporal_entropy,
                indeterminacy,assigned_event_id,event_label,synthetic,
                resolved_status,record_confidence,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (str(_uuid.uuid4()), series_id, run_id,
             _sid, prior_row["slot_label"], prior_row["slot_polarity"], prior_row["slot_kind"],
             prior_row["probability"], prior_row["temporal_energy"], prior_row["temporal_coherence"],
             prior_row["temporal_entropy"], prior_row["indeterminacy"], prior_row["assigned_event_id"],
             prior_row["event_label"], prior_row["synthetic"], prior_row["resolved_status"],
             prior_row["record_confidence"], now)
        )
        captured += 1
        print(f"[Phase 6] Slot '{_sid}' unmapped this observation — carried forward "
              f"previous branch_probability (delta=0.0)")

    print(f"[Phase 6] Captured {captured} slot snapshots for run {run_id}")

    # Field deltas vs previous run in series
    prev = conn.execute(
        """SELECT id FROM runs WHERE series_id=? AND id!=?
           AND run_index IS NOT NULL ORDER BY run_index DESC LIMIT 1""",
        (series_id, run_id)
    ).fetchone()
    if prev:
        prev_map = {r["slot_id"]: r for r in conn.execute(
            "SELECT slot_id,probability,temporal_energy,indeterminacy FROM slot_snapshots WHERE run_id=?",
            (prev["id"],)
        ).fetchall()}
        # Phase 6.7.2: fallback map by slot_label, for series whose history
        # spans the canonical_slot_id carry-through fix — the previous run may
        # have recorded this outcome under an old-format slot_id (the label
        # text itself, used as a fallback before the fix), while this run uses
        # the proper slot_id. slot_label has been stable throughout. This
        # heals the seam on this one observation; the resulting field_deltas
        # row is stored under THIS run's (correct) slot_id, so subsequent
        # observations match directly via slot_id with no fallback needed.
        prev_map_by_label = {r["slot_label"]: r for r in conn.execute(
            "SELECT slot_label,probability,temporal_energy,indeterminacy FROM slot_snapshots WHERE run_id=?",
            (prev["id"],)
        ).fetchall()}
        for snap in conn.execute(
            "SELECT slot_id,slot_label,probability,temporal_energy,indeterminacy FROM slot_snapshots WHERE run_id=?",
            (run_id,)
        ).fetchall():
            sid = snap["slot_id"]
            p = prev_map.get(sid) or prev_map_by_label.get(snap["slot_label"])
            if p:
                pd = float(snap["probability"] or 0) - float(p["probability"] or 0)
                ed = float(snap["temporal_energy"] or 0) - float(p["temporal_energy"] or 0)
                id_= float(snap["indeterminacy"] or 0) - float(p["indeterminacy"] or 0)
                # Phase 7.4: compute slot-level field-continuity resonance.
                # Resonance measures how coherent the observed movement is with
                # the previous trajectory: expected_delta = prev_delta * inertia.
                # High resonance → field evolving coherently.
                # Low resonance → unexpected shift (new evidence, structural break).
                try:
                    _prev_fd = conn.execute(
                        """SELECT probability_delta FROM field_deltas
                           WHERE series_id=? AND slot_id=? AND to_run_id=?
                           ORDER BY created_at DESC LIMIT 1""",
                        (series_id, sid, prev["id"])
                    ).fetchone()
                    _prev_delta   = float(_prev_fd["probability_delta"] or 0.0) if _prev_fd else 0.0
                    _expected     = _prev_delta * 0.5   # inertia_factor = 0.5
                    _res_error    = abs(pd - _expected)
                    _res_score    = round(max(0.0, 1.0 - _res_error / 0.25), 3)
                except Exception:
                    _res_score = None

                try:
                    conn.execute(
                        """INSERT OR REPLACE INTO field_deltas
                           (id,series_id,from_run_id,to_run_id,slot_id,
                            probability_delta,energy_delta,indeterminacy_delta,
                            summary,resonance_score,created_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (str(_uuid.uuid4()), series_id, prev["id"], run_id, sid,
                         pd, ed, id_, f"delta p={pd:+.3f}", _res_score, now)
                    )
                except Exception:
                    # Fallback: resonance_score column may not exist on old DBs
                    conn.execute(
                        """INSERT OR REPLACE INTO field_deltas
                           (id,series_id,from_run_id,to_run_id,slot_id,
                            probability_delta,energy_delta,indeterminacy_delta,
                            summary,created_at)
                           VALUES (?,?,?,?,?,?,?,?,?,?)""",
                        (str(_uuid.uuid4()), series_id, prev["id"], run_id, sid,
                         pd, ed, id_, f"delta p={pd:+.3f}", now)
                    )

def health():
    return {"status": "ok", "live_mode": LIVE_MODE}
