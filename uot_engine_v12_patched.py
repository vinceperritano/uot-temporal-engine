"""
UOT Temporal Extrapolation Engine — v0.4

New in v0.4:
  - SeededEvent and SourceRef data structures
  - Five UOT field estimation functions (indeterminacy, coherence, entropy, energy, salience)
  - estimate_uot_fields() converts raw candidate data into full EventNode parameters
  - seed_graph_from_topic() pipeline: collect → extract → infer branches → infer edges → estimate
  - User review step is modeled as an explicit observation event
  - All auto-estimated fields are flagged as provisional; user adjustments are tracked
  - Every EventNode carries source metadata and confidence notes

Everything from v0.3 is preserved unchanged.
v0.4 adds the seeding layer on top.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import math


# ============================================================
# Stage 0: Model Parameters
# ============================================================

@dataclass
class ModelParams:
    w_temporal_energy: float  = 1.0
    w_indeterminacy: float    = 1.0
    w_causal_conflict: float  = 1.0
    w_entropy: float          = 1.0
    w_coherence: float        = 1.0
    gamma: float       = 0.1
    alpha: float       = 0.5
    beta: float        = 0.5
    k_temporal: float  = 1.0
    min_value: float                    = 0.0
    max_value: float                    = 1.0
    min_residual_temporal_energy: float = 0.05
    instability_threshold: float        = 0.65


def clamp(value: float, min_v: float = 0.0, max_v: float = 1.0) -> float:
    return max(min_v, min(value, max_v))


# ============================================================
# v0.4 NEW: Source and Seeded Event Structures
# ============================================================

@dataclass
class SourceRef:
    """A source that supports an event node's existence and probability estimate."""
    title: str
    publisher: str
    url: str = ""
    date: str = ""
    relevance: float = 0.5    # 0-1: how directly relevant to this event
    stance: str = "neutral"   # "supporting" | "neutral" | "contradicting" | "uncertain"

@dataclass
class SourcePacket:
    """
    A compressed, structured observational record built from a raw source.
    Stage A receives SourcePackets, not raw article text.
    Full text retained in raw_text_ref for fallback/audit.
    """
    title: str
    publisher: str
    url: str   = ""
    date: str  = ""
    source_type: str   = "article"
    credibility: float = 0.5
    recency: float     = 0.5
    summary: str = ""
    key_claims: List[str]        = field(default_factory=list)
    causal_phrases: List[str]    = field(default_factory=list)
    branch_phrases: List[str]    = field(default_factory=list)
    evidence_snippets: List[str] = field(default_factory=list)
    raw_text_ref: str  = ""
    raw_text_chars: int = 0


@dataclass
class SearchResult:
    """Single result from one search query. Tracks which query found it."""
    query: str
    purpose: str
    title: str
    publisher: str
    url: str
    date: str  = ""
    snippet: str = ""
    source_type_hint: str = "article"
    rank: int  = 0


@dataclass
class RawDocument:
    """Fetched document with full text and cache key for raw_text_ref."""
    title: str
    publisher: str
    url: str
    date: str
    text: str
    metadata: dict
    cache_key: str



@dataclass
class ExtractionDiagnostics:
    """
    AI extractor rationale fields — one per SeededEvent.
    Lets the user-observer review WHY each score was assigned before simulation.
    Stored per event; displayed in the review step.
    """
    downstream_impact_rationale: str = ""
    disruption_score_rationale:  str = ""
    novelty_rationale:           str = ""
    probability_rationale:       str = ""
    causal_rationale:            str = ""
    branch_rationale:            str = ""
    reconciliation_flags:        str = ""
    confidence:                  float = 0.5

@dataclass
class SeededEvent:
    """
    Raw candidate event from the seeding pipeline.
    Contains all inputs needed by estimate_uot_fields().
    UOT fields are NOT set here — they are derived by estimation.
    """
    id: str
    label: str
    description: str

    temporal_status: str = "unresolved"  # resolved | active | unresolved | counterfactual
    probability: float   = 0.5
    time_estimate: Optional[float] = None
    time_uncertainty: float = 0.5

    categories: Dict[str, float] = field(default_factory=dict)
    sources: List[SourceRef]     = field(default_factory=list)

    # Computed from sources
    source_count: int      = 0
    source_agreement: float = 0.5   # 0=total disagreement, 1=full agreement
    recency: float          = 0.5   # 0=old, 1=very recent

    branch_group: Optional[str]  = None
    branch_label: Optional[str]  = None

    # Causal candidate IDs (other SeededEvent IDs this might cause/follow)
    causal_candidates: List[str] = field(default_factory=list)

    # Signals for UOT estimation
    downstream_impact: float  = 0.5  # how many/large are downstream consequences
    disruption_score: float   = 0.5  # social/systemic disruption level
    novelty: float            = 0.5  # how surprising/unprecedented
    causal_support: float     = 0.5  # how strongly supported by causal chain
    extraction_notes: Optional[ExtractionDiagnostics] = None  # v0.9: AI extractor rationale


# ============================================================
# v0.4 NEW: UOT Field Estimation Functions
# ============================================================

def estimate_indeterminacy(
    probability: float,
    time_uncertainty: float,
    source_agreement: float,
    branch_group: Optional[str]
) -> float:
    """
    Indeterminacy = degree of unresolved possibility.

    High when:
      - probability near 0.5 (neither confirmed nor ruled out)
      - timing is vague (high time_uncertainty)
      - sources disagree
      - event is part of a mutually exclusive branch group

    Faithful to UOT: indeterminacy is unresolved possibility,
    not merely ignorance.
    """
    probability_uncertainty   = 1.0 - abs(probability - 0.5) * 2.0
    source_disagreement       = 1.0 - source_agreement
    branch_unresolvedness     = 1.0 if branch_group is not None else 0.5

    return clamp(
        0.35 * time_uncertainty
        + 0.25 * probability_uncertainty
        + 0.20 * source_disagreement
        + 0.20 * branch_unresolvedness
    )


def estimate_record_coherence(
    source_agreement: float,
    source_count: int,
    recency: float
) -> float:
    """
    Record coherence = stability and consistency of the observation record.

    This is epistemic/observational coherence — not temporal/system coherence.
    Multiple observers agreeing stabilizes the record of an event.
    It does not necessarily stabilize the event's role in the timeline.

    Faithful to UOT: observer agreement stabilizes realized informational structure,
    but chaotic events remain temporally unstable even when well-observed.
    """
    source_count_factor = min(source_count / 5.0, 1.0)

    return clamp(
        0.60 * source_agreement
        + 0.25 * source_count_factor
        + 0.15 * recency
    )


def estimate_temporal_coherence(
    record_coherence: float,
    causal_support: float,
    time_uncertainty: float,
    temporal_status: str,
    disruption_score: float,
    temporal_entropy: float
) -> float:
    """
    Temporal coherence = stability of the event within the causal-temporal structure.

    v0.6: source_agreement is gone from this function entirely.
    It now lives in record_coherence, which is estimated separately.

    record_coherence contributes only through a disruption-filtered term:
    observers can agree on a chaotic event, but that agreement does not
    make the timeline node stable.

    temporal_entropy is a new negative term: high entropy events resist coherence
    even if disruption is moderate.

    In UOT: observer agreement stabilizes the record.
    It does not stabilize the event's temporal role.
    """
    temporal_specificity = 1.0 - time_uncertainty

    # v0.6: lower status ceilings — status alone cannot guarantee high coherence
    status_stability_map = {
        "resolved":       0.65,
        "active":         0.55,
        "unresolved":     0.45,
        "counterfactual": 0.35
    }
    status_stability = status_stability_map.get(temporal_status, 0.5)

    # Record coherence contributes only when disruption is low
    record_stability = record_coherence * (1.0 - disruption_score)

    return clamp(
        0.35 * record_stability
        + 0.25 * causal_support
        + 0.20 * temporal_specificity
        + 0.15 * status_stability
        - 0.10 * temporal_entropy   # v0.7: reduced from 0.15
    )


def estimate_temporal_entropy(
    downstream_impact: float,
    source_agreement: float,
    disruption_score: float,
    branch_group: Optional[str],
    novelty: float
) -> float:
    """
    Temporal entropy = degree to which an event disperses uncertainty,
    conflict, or irreversible consequences into the timeline.

    High entropy events create downstream disorder, institutional stress,
    information fragmentation, or many second-order consequences.

    Faithful to UOT: entropy, decoherence, and temporal asymmetry
    are tied to information becoming harder to reverse or reassemble.
    """
    source_disagreement   = 1.0 - source_agreement
    branch_unresolvedness = 1.0 if branch_group is not None else 0.4

    return clamp(
        0.30 * downstream_impact
        + 0.25 * source_disagreement
        + 0.20 * disruption_score
        + 0.15 * branch_unresolvedness
        + 0.10 * novelty
    )


def estimate_initial_temporal_energy(
    indeterminacy: float,
    temporal_entropy: float,
    downstream_impact: float,
    observer_salience: float,
    branch_pressure: float,
    temporal_coherence: float,
    temporal_status: str
) -> float:
    """
    Temporal energy = intensity of change-pressure at this event node.
    Estimated last, after the other UOT fields are known.

    Status boost ensures resolved events retain some energy (causal inertia)
    while counterfactuals have reduced energy.
    """
    base = (
        0.30 * indeterminacy
        + 0.25 * temporal_entropy
        + 0.20 * downstream_impact
        + 0.15 * observer_salience
        + 0.10 * branch_pressure
        - 0.20 * temporal_coherence
    )

    status_boost = {
        "resolved":       0.10,
        "active":         0.25,
        "unresolved":     0.20,
        "counterfactual": -0.10
    }.get(temporal_status, 0.0)

    return clamp(base + status_boost)


def estimate_observer_salience_from_categories(
    categories: Dict[str, float],
    observer_measurement_basis: Dict[str, float]
) -> float:
    """
    Observer salience = how much this event aligns with
    the observer's measurement basis (interpretive categories).
    """
    if not categories or not observer_measurement_basis:
        return 0.5

    alignment = sum(
        categories.get(cat, 0.0) * weight
        for cat, weight in observer_measurement_basis.items()
    )

    return clamp(alignment / max(len(observer_measurement_basis), 1))



# ============================================================
# v0.8: Causal Support Estimation — Hybrid Graph + Source
# ============================================================

def estimate_graph_upstream_support(
    target_id: str,
    graph: 'EventGraph'
) -> float:
    """
    Structural causal support from upstream nodes.
    Weighted by source probability, edge causal weight, certainty, and relation type.
    Maps the possible signed range into 0-1.
    """
    incoming = graph.incoming_edges(target_id)
    if not incoming:
        return 0.5

    support_terms = []
    for edge in incoming:
        source   = graph.nodes[edge.source_id]
        certainty = 1.0 - edge.uncertainty

        if edge.relation_type in ("inhibitory", "exclusive"):
            sign = -1.0
        elif edge.relation_type == "reinforcing":
            sign = 1.2
        else:
            sign = 1.0

        support_terms.append(
            source.probability * edge.causal_weight * certainty * sign
        )

    raw = sum(support_terms) / max(len(support_terms), 1)
    return clamp((raw + 1.0) / 2.0)   # map [-1,1] → [0,1]


def estimate_source_causal_evidence(candidate_causal_support: float) -> float:
    """
    Placeholder for source-text causal extraction in live pipeline.

    In v0.8 live seeding, this is replaced by an extraction model that
    identifies explicit causal language ('due to', 'led to', 'triggered by')
    in source texts and returns a SourceCausalClaim list with strengths.

    For now, passes through the manually supplied causal_support value
    or 0.5 if none supplied.
    """
    return clamp(candidate_causal_support)


def estimate_temporal_order_consistency(
    target: 'EventNode',
    incoming_edges: list,
    graph: 'EventGraph'
) -> float:
    """
    Checks whether causes precede effects in the proposed temporal ordering.
    Causes with undefined time_estimate are treated as neutral (0.5).
    Feedback/anticipatory edges get partial credit even if they appear later.
    """
    if not incoming_edges or target.time_estimate is None:
        return 0.5

    scores = []
    for edge in incoming_edges:
        source = graph.nodes[edge.source_id]
        if source.time_estimate is None:
            scores.append(0.5)
        elif source.time_estimate <= target.time_estimate:
            scores.append(1.0)
        elif edge.feedback_strength > 0.0:
            scores.append(0.7)    # feedback edges permitted even if later
        else:
            scores.append(0.2)    # temporally inconsistent cause

    return clamp(sum(scores) / len(scores))


def estimate_path_coherence(
    target_id: str,
    graph: 'EventGraph'
) -> float:
    """
    Checks whether upstream sources are compatible with the target's causal pathway.

    Branch conflicts (same branch_group, different branch_label, non-inhibitory edge)
    indicate a causal graph that mixes incompatible futures — penalized.
    Compatible causes on reinforcing or causal edges score highly.
    """
    incoming = graph.incoming_edges(target_id)
    if not incoming:
        return 0.5

    target = graph.nodes[target_id]
    scores = []

    for edge in incoming:
        source = graph.nodes[edge.source_id]

        if (source.branch_group is not None
                and source.branch_group == target.branch_group
                and source.branch_label != target.branch_label):
            # Same branch group, different labels = mutually exclusive
            if edge.relation_type in ("inhibitory", "exclusive"):
                scores.append(1.0)   # correctly modeled conflict
            else:
                scores.append(0.0)   # incorrectly mixing exclusive futures
        else:
            if edge.relation_type in ("causal", "reinforcing", "enabling"):
                scores.append(1.0)
            elif edge.relation_type in ("inhibitory", "exclusive"):
                scores.append(0.6)   # inhibitory edges are coherent, just limiting
            else:
                scores.append(0.5)

    return clamp(sum(scores) / len(scores))


def estimate_causal_support_from_graph(
    target_id: str,
    graph: 'EventGraph',
    source_causal_evidence: float = 0.5
) -> float:
    """
    Composite causal support from four terms.

    In UOT: causal_support = the degree to which an event is supported by
    the resolved and active causal structure of the timeline — how strongly
    the timeline 'constrains' or 'wants' this event given prior observations,
    causal edges, source evidence, temporal order, and branch compatibility.

    graph_upstream_support    (0.35) — structural graph evidence
    source_causal_evidence    (0.30) — source-text causal claims
    temporal_order_consistency (0.20) — cause precedes effect
    path_coherence            (0.15) — no branch conflicts
    """
    incoming = graph.incoming_edges(target_id)
    target   = graph.nodes[target_id]

    gus = estimate_graph_upstream_support(target_id, graph)
    toc = estimate_temporal_order_consistency(target, incoming, graph)
    pc  = estimate_path_coherence(target_id, graph)

    return clamp(
        0.35 * gus
        + 0.30 * source_causal_evidence
        + 0.20 * toc
        + 0.15 * pc
    )


def compute_source_agreement(sources: List[SourceRef]) -> float:
    """
    Estimates agreement level from source stances.
    All supporting = 1.0; mixed = 0.5; all contradicting = 0.0.
    """
    if not sources:
        return 0.5

    stance_scores = {"supporting": 1.0, "neutral": 0.6, "uncertain": 0.4, "contradicting": 0.0}
    scores = [stance_scores.get(s.stance, 0.5) * s.relevance for s in sources]
    weights = [s.relevance for s in sources]

    if sum(weights) == 0:
        return 0.5

    return clamp(sum(scores) / sum(weights))


# ============================================================
# v0.4 NEW: Full Field Estimation Pipeline
# ============================================================

def estimate_uot_fields(
    candidate: SeededEvent,
    observer_measurement_basis: Dict[str, float]
) -> dict:
    """
    Converts a SeededEvent into a complete set of UOT field estimates.
    Returns a dict that can be used to construct an EventNode.

    All estimated fields are flagged as provisional.
    """
    probability      = candidate.probability
    time_uncertainty = candidate.time_uncertainty
    temporal_status  = candidate.temporal_status
    branch_group     = candidate.branch_group

    # Recompute source_agreement from source objects if available
    source_agreement = (
        compute_source_agreement(candidate.sources)
        if candidate.sources
        else candidate.source_agreement
    )

    observer_salience = estimate_observer_salience_from_categories(
        candidate.categories,
        observer_measurement_basis
    )

    branch_pressure = 1.0 if branch_group is not None else 0.3

    indeterminacy = estimate_indeterminacy(
        probability=probability,
        time_uncertainty=time_uncertainty,
        source_agreement=source_agreement,
        branch_group=branch_group
    )

    # v0.6: estimate record_coherence first; temporal_coherence uses it as input
    record_coherence_val = estimate_record_coherence(
        source_agreement=source_agreement,
        source_count=len(candidate.sources) or candidate.source_count,
        recency=candidate.recency
    )

    temporal_entropy_est = estimate_temporal_entropy(
        downstream_impact=candidate.downstream_impact,
        source_agreement=source_agreement,
        disruption_score=candidate.disruption_score,
        branch_group=branch_group,
        novelty=candidate.novelty
    )

    temporal_coherence = estimate_temporal_coherence(
        record_coherence=record_coherence_val,
        causal_support=candidate.causal_support,
        time_uncertainty=time_uncertainty,
        temporal_status=temporal_status,
        disruption_score=candidate.disruption_score,
        temporal_entropy=temporal_entropy_est
    )

    temporal_entropy = temporal_entropy_est  # v0.6: already computed above

    temporal_energy = estimate_initial_temporal_energy(
        indeterminacy=indeterminacy,
        temporal_entropy=temporal_entropy,
        downstream_impact=candidate.downstream_impact,
        observer_salience=observer_salience,
        branch_pressure=branch_pressure,
        temporal_coherence=temporal_coherence,
        temporal_status=temporal_status
    )

    confidence_note = (
        f"Auto-estimated from {len(candidate.sources)} sources "
        f"(agreement: {source_agreement:.2f}, recency: {candidate.recency:.2f}). "
        f"Provisional — review before simulation."
    )

    return {
        "id":                  candidate.id,
        "label":               candidate.label,
        "probability":         probability,
        "time_estimate":       candidate.time_estimate,
        "time_uncertainty":    time_uncertainty,
        "temporal_energy":     round(temporal_energy,    3),
        "temporal_coherence":  round(temporal_coherence, 3),
        "temporal_entropy":    round(temporal_entropy,   3),
        "indeterminacy":       round(indeterminacy,      3),
        "observer_sensitivity":round(observer_salience,  3),
        "categories":          candidate.categories,
        "branch_group":        branch_group,
        "branch_label":        candidate.branch_label,
        "temporal_status":     temporal_status,
        # Provenance
        "source_count":        len(candidate.sources) or candidate.source_count,
        "source_agreement":    round(source_agreement, 3),
        "record_coherence":    round(record_coherence_val, 3),   # v0.6
        "confidence_note":     confidence_note,
        # User adjustment tracking
        "disruption_score":    candidate.disruption_score,   # v0.5: propagated to EventNode
        "downstream_impact":   candidate.downstream_impact,  # v0.8: passed to metadata for recompute
        "source_causal_evidence": candidate.causal_support,  # v0.12-patch: original source evidence
        # preserved separately from graph-derived causal_support_graph
        "auto_estimated": {
            "temporal_energy":     round(temporal_energy,    3),
            "temporal_coherence":  round(temporal_coherence, 3),
            "temporal_entropy":    round(temporal_entropy,   3),
            "indeterminacy":       round(indeterminacy,      3),
            "record_coherence":    round(record_coherence_val, 3),
        },
        "user_adjusted": {}
    }


# ============================================================
# v0.4 NEW: Seeding Pipeline (stub for live data; test data below)
# ============================================================

def infer_causal_edges_from_candidates(
    seeded_events: List[SeededEvent]
) -> List[dict]:
    """
    Infers candidate causal edges from causal_candidates lists.
    Returns edge specs for user review before building the graph.

    In v0.4 with live data, this would also be informed by
    AI-extracted causal language from source texts.
    """
    edge_candidates = []
    id_set = {e.id for e in seeded_events}

    for event in seeded_events:
        for target_id in event.causal_candidates:
            if target_id in id_set:
                edge_candidates.append({
                    "source_id": event.id,
                    "target_id": target_id,
                    "causal_weight": 0.5,    # provisional
                    "uncertainty": 0.5,      # provisional
                    "relation_type": "causal",
                    "provisional": True
                })

    return edge_candidates


def print_review_summary(
    estimated_nodes: List[dict],
    edge_candidates: List[dict],
    seeded_events = None
) -> None:
    """
    Displays the provisional graph for user-observer review.
    In v0.4 this is printed; in a future UI version it becomes interactive.

    The review step is modeled as an explicit UOT observation event:
    the observer's measurement basis and interpretive frame shape
    which events become meaningful within the timeline structure.
    """
    diag_lookup = {}
    if seeded_events:
        for se in seeded_events:
            if hasattr(se, 'extraction_notes') and se.extraction_notes:
                diag_lookup[se.id] = se.extraction_notes

    print("\n" + "="*70)
    print("PROVISIONAL GRAPH — OBSERVER REVIEW REQUIRED BEFORE SIMULATION")
    print("v0.12: Stage A received compressed SourcePackets, not raw article text")
    print("="*70)
    print("Review each event. Adjust probability and UOT fields if needed.")
    print("Confirm or remove causal edges. Add missing events.\n")

    for node in estimated_nodes:
        status_tag = f"[{node['temporal_status'][:3].upper()}]"
        bg_flag    = f" [BRANCH: {node['branch_group']}]" if node['branch_group'] else ""
        print(f"  {node['id']} {status_tag}{bg_flag}")
        print(f"  Label: {node['label']}")
        print(f"  P={node['probability']:.2f}  "
              f"TE={node['temporal_energy']:.3f}  "
              f"TCoh={node['temporal_coherence']:.3f}  "
              f"RCoh={node.get('record_coherence', '?'):.3f}  "
              f"Ent={node['temporal_entropy']:.3f}  "
              f"Indet={node['indeterminacy']:.3f}  "
              f"ObsSens={node['observer_sensitivity']:.3f}")
        print(f"  Sources: {node['source_count']}  Agreement: {node['source_agreement']:.2f}")
        diag = diag_lookup.get(node["id"])
        if diag:
            print(f"  [EXTRACTOR confidence={diag.confidence:.2f}]")
            if diag.downstream_impact_rationale:
                print(f"    DI:  {diag.downstream_impact_rationale[:88]}")
            if diag.disruption_score_rationale:
                print(f"    DS:  {diag.disruption_score_rationale[:88]}")
            if diag.novelty_rationale:
                print(f"    NV:  {diag.novelty_rationale[:88]}")
            if diag.branch_rationale:
                print(f"    BG:  {diag.branch_rationale[:88]}")
            if diag.causal_rationale:
                print(f"    CAU: {diag.causal_rationale[:88]}")
            if diag.reconciliation_flags and diag.reconciliation_flags not in ("","None."):
                print(f"    [FLAG] {diag.reconciliation_flags[:88]}")
        else:
            print(f"  {node['confidence_note']}")
        if node.get("user_adjusted"):
            print(f"  USER ADJUSTMENTS: {node['user_adjusted']}")
        print()

    print("Causal edges (Stage C inferred):")
    for edge in edge_candidates:
        flag = " [PROVISIONAL]" if edge.get("provisional") else " [Stage C]"
        rationale = edge.get("stage_c_rationale", "")
        rat_str = f" — {rationale[:60]}" if rationale else ""
        print(f"  {edge['source_id']} --[{edge['relation_type']}, w={edge['causal_weight']:.2f}]--> "
              f"{edge['target_id']}{flag}{rat_str}")

    print("\n[In a live UI, the observer edits this graph before simulation runs.]")
    print("="*70)


# ============================================================
# v0.3 Core Structures (preserved)
# ============================================================

@dataclass
class EvidenceEvent:
    id: str
    target_node_id: str
    likelihood_ratio: float
    confidence: float = 1.0
    timestamp: Optional[float] = None
    applied: bool = False
    source: Optional[str] = None
    description: str = ""


@dataclass
class EventNode:
    id: str
    label: str
    probability: float = 0.5
    time_estimate: Optional[float] = None
    time_uncertainty: float = 0.5
    temporal_energy: float   = 0.5
    temporal_coherence: float = 0.5
    temporal_entropy: float  = 0.5
    indeterminacy: float     = 0.5
    observer_sensitivity: float = 0.5
    categories: Dict[str, float] = field(default_factory=dict)
    branch_group: Optional[str] = None
    branch_label: Optional[str] = None
    temporal_status: str = "unresolved"
    disruption_score: float = 0.5   # v0.5: inherent event disruption level
    record_coherence: float = 0.5   # v0.6: epistemic/observational coherence (separate from temporal)
    # v0.4: provenance fields
    source_count: int = 0
    source_agreement: float = 0.5
    confidence_note: str = ""
    auto_estimated: Dict[str, float] = field(default_factory=dict)
    user_adjusted: Dict[str, float]  = field(default_factory=dict)
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class EventEdge:
    source_id: str
    target_id: str
    causal_weight: float    = 0.5
    delay: float            = 1.0
    uncertainty: float      = 0.5
    feedback_strength: float = 0.0
    relation_type: str      = "causal"


@dataclass
class EventGraph:
    nodes: Dict[str, EventNode] = field(default_factory=dict)
    edges: List[EventEdge] = field(default_factory=list)

    def add_node(self, node: EventNode) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: EventEdge) -> None:
        self.edges.append(edge)

    def incoming_edges(self, node_id: str) -> List[EventEdge]:
        return [e for e in self.edges if e.target_id == node_id]

    def outgoing_edges(self, node_id: str) -> List[EventEdge]:
        return [e for e in self.edges if e.source_id == node_id]


@dataclass
class ObserverState:
    id: str
    label: str
    information_state: Dict[str, float]  = field(default_factory=dict)
    attention_state: Dict[str, float]    = field(default_factory=dict)
    measurement_basis: Dict[str, float]  = field(default_factory=dict)
    intention_state: Dict[str, float]    = field(default_factory=dict)
    action_capacity: Dict[str, float]    = field(default_factory=dict)
    coherence_level: float   = 0.5
    coupling_strength: float = 0.5


@dataclass
class TemporalState:
    tau_rate: float              = 1.0
    temporal_flux: float         = 0.5
    temporal_coherence: float    = 0.5
    temporal_entropy: float      = 0.5
    indeterminacy_density: float = 0.5
    causal_curvature: float      = 0.5
    branch_potential: float      = 0.5
    observer_coupling: float     = 0.5


@dataclass
class WorldTimelineState:
    event_graph: EventGraph
    observers: List[ObserverState]       = field(default_factory=list)
    evidence_events: List[EvidenceEvent] = field(default_factory=list)
    temporal_state: TemporalState        = field(default_factory=TemporalState)
    step: int = 0


# ============================================================
# v0.4 NEW: Build EventGraph from estimated nodes + reviewed edges
# ============================================================

def build_graph_from_estimates(
    estimated_nodes: List[dict],
    edge_specs: List[dict]
) -> EventGraph:
    """
    Builds an EventGraph from the output of estimate_uot_fields()
    and the reviewed edge specifications.
    """
    graph = EventGraph()

    for n in (x for x in estimated_nodes if isinstance(x, dict)):
        node = EventNode(
            id               = n["id"],
            label            = n["label"],
            probability      = n["probability"],
            time_estimate    = n.get("time_estimate"),
            time_uncertainty = n["time_uncertainty"],
            temporal_energy  = n["temporal_energy"],
            temporal_coherence = n["temporal_coherence"],
            temporal_entropy = n["temporal_entropy"],
            indeterminacy    = n["indeterminacy"],
            observer_sensitivity = n["observer_sensitivity"],
            categories       = n.get("categories", {}),
            branch_group     = n.get("branch_group"),
            branch_label     = n.get("branch_label"),
            temporal_status  = n.get("temporal_status", "unresolved"),
            disruption_score = n.get("disruption_score", 0.5),   # v0.5
            record_coherence = n.get("record_coherence", 0.5),   # v0.6
            source_count     = n.get("source_count", 0),
            source_agreement = n.get("source_agreement", 0.5),
            confidence_note  = n.get("confidence_note", ""),
            auto_estimated   = n.get("auto_estimated", {}),
            user_adjusted    = n.get("user_adjusted", {}),
            metadata         = {
                "downstream_impact":       str(n.get("downstream_impact", 0.5)),
                "source_causal_evidence":  str(n.get("source_causal_evidence", 0.5)),
                # source_causal_evidence is the original SeededEvent.causal_support,
                # kept separate from graph-derived causal_support_graph so that
                # recompute_coherence_and_energy() never feeds back on itself.
            },
        )
        graph.add_node(node)

    for e in edge_specs:
        graph.add_edge(EventEdge(
            source_id      = e["source_id"],
            target_id      = e["target_id"],
            causal_weight  = e.get("causal_weight", 0.5),
            uncertainty    = e.get("uncertainty", 0.5),
            relation_type  = e.get("relation_type", "causal"),
            feedback_strength = e.get("feedback_strength", 0.0)
        ))

    return graph


# ============================================================
# v0.3 Simulation Logic (preserved, unchanged)
# ============================================================

def apply_likelihood_update(node: EventNode, likelihood_ratio: float, confidence: float = 1.0) -> None:
    collapse_resistance = clamp(0.5 * node.indeterminacy + 0.5 * node.time_uncertainty)
    effective_lr = 1.0 + (likelihood_ratio - 1.0) * confidence * (1.0 - collapse_resistance)
    effective_lr = max(effective_lr, 0.01)
    raw_new = node.probability * effective_lr
    denom   = raw_new + (1.0 - node.probability)
    if denom > 0:
        node.probability = clamp(raw_new / denom)


def apply_pending_evidence(world: WorldTimelineState) -> WorldTimelineState:
    for ev in world.evidence_events:
        if ev.applied:
            continue
        if ev.target_node_id not in world.event_graph.nodes:
            continue
        apply_likelihood_update(world.event_graph.nodes[ev.target_node_id], ev.likelihood_ratio, ev.confidence)
        ev.applied = True
    return world


def normalize_branch_groups(world: WorldTimelineState) -> None:
    groups: Dict[str, List[EventNode]] = {}
    for node in world.event_graph.nodes.values():
        if node.branch_group is not None:
            groups.setdefault(node.branch_group, []).append(node)
    for group_nodes in groups.values():
        total = sum(max(n.probability, 1e-9) for n in group_nodes)
        if total > 0:
            for node in group_nodes:
                node.probability = clamp(node.probability / total)


def compute_salience(node: EventNode, observers: List[ObserverState], params: ModelParams) -> float:
    total = 0.0
    for obs in observers:
        direct         = obs.attention_state.get(node.id, 0.0)
        basis          = sum(node.categories.get(c, 0.0) * w for c, w in obs.measurement_basis.items())
        combined_focus = clamp((direct + basis) / 2.0)
        total         += obs.coupling_strength * node.observer_sensitivity * combined_focus
    return params.alpha * total


def compute_temporal_energy_sources(world, dt, params):
    graph = world.event_graph
    next_energies = {}
    for node_id, node in graph.nodes.items():
        salience = compute_salience(node, world.observers, params)
        if node.temporal_status == "resolved":
            obs_src = salience * 0.1;  indet_src = node.indeterminacy * 0.1
            coh_d   = node.temporal_coherence * 0.6; ent_d = node.temporal_entropy * 0.4
        elif node.temporal_status == "active":
            obs_src = salience * 0.8;  indet_src = node.indeterminacy * 0.6
            coh_d   = node.temporal_coherence * 0.3; ent_d = node.temporal_entropy * 0.2
        elif node.temporal_status == "counterfactual":
            obs_src = salience * 0.05; indet_src = 0.0
            coh_d   = node.temporal_coherence * 0.8; ent_d = node.temporal_entropy * 0.5
        else:
            realization = 1.0 - (node.probability * 0.5)
            obs_src = salience * realization; indet_src = node.indeterminacy * 0.5
            coh_d   = node.temporal_coherence * 0.4; ent_d = node.temporal_entropy * 0.3
        external = float(node.metadata.get("external_source", 0.0))
        dTE  = (obs_src + indet_src + external - coh_d - ent_d) * dt
        floor = params.min_residual_temporal_energy if (node.temporal_status == "resolved" or node.probability >= 0.9) else params.min_value
        next_energies[node_id] = clamp(node.temporal_energy + dTE, floor, params.max_value)
    for nid, val in next_energies.items():
        graph.nodes[nid].temporal_energy = val
    return world


def propagate_temporal_flux(world, dt, params):
    graph = world.event_graph
    e_d = {nid: 0.0 for nid in graph.nodes}
    i_d = {nid: 0.0 for nid in graph.nodes}
    for edge in graph.edges:
        src  = graph.nodes[edge.source_id]
        base = edge.causal_weight * src.temporal_energy * dt
        cert = 1.0 - edge.uncertainty
        if edge.relation_type == "inhibitory":
            e_d[edge.target_id] -= base * cert * 0.5
        elif edge.relation_type == "reinforcing":
            e_d[edge.target_id] += base * cert * 1.2
        else:
            e_d[edge.target_id] += base * cert
        i_d[edge.target_id] += base * edge.uncertainty * 0.3
        if edge.feedback_strength > 0.0:
            tgt = graph.nodes[edge.target_id]
            e_d[edge.source_id] += edge.feedback_strength * tgt.temporal_energy * dt * 0.5
    for nid, node in graph.nodes.items():
        floor = params.min_residual_temporal_energy if (node.temporal_status == "resolved" or node.probability >= 0.9) else params.min_value
        node.temporal_energy = clamp(node.temporal_energy + e_d[nid] * 0.15, floor, params.max_value)
        node.indeterminacy   = clamp(node.indeterminacy   + i_d[nid])
    energies = [n.temporal_energy for n in graph.nodes.values()]
    mean_e   = sum(energies) / len(energies)
    var_e    = sum((e - mean_e) ** 2 for e in energies) / len(energies)
    world.temporal_state.causal_curvature = clamp(math.sqrt(var_e) * 2.0)
    world.temporal_state.temporal_flux    = mean_e
    return world



def coherence_ceiling(node) -> float:
    """
    v0.6: Three-way ceiling on temporal coherence during simulation.

    Observer attention can raise coherence toward the ceiling,
    but three independent constraints limit how high it can go:

      status_ceiling:    resolved events can be more stable than unresolved
      disruption_ceiling: inherently disruptive events resist full coherence
      entropy_ceiling:   high-entropy events resist coherence even if disruption is moderate

    The lowest ceiling wins — most restrictive constraint applies.

    Faithful to UOT: observation clarifies the record; it does not
    fully stabilize a disruptive or entropy-generating causal node.
    """
    status_ceiling = {
        "resolved":       0.70,
        "active":         0.65,
        "unresolved":     0.60,
        "counterfactual": 0.45
    }.get(node.temporal_status, 0.6)

    disruption_ceiling = 1.0 - 0.65 * node.disruption_score
    entropy_ceiling    = 1.0 - 0.45 * node.temporal_entropy

    return clamp(min(status_ceiling, disruption_ceiling, entropy_ceiling))


def apply_observer_effect(world, dt, params):
    graph = world.event_graph
    observers = world.observers
    avg_coh = sum(o.coherence_level for o in observers) / len(observers) if observers else 0.0
    for node_id, node in graph.nodes.items():
        total_salience = 0.0; total_action = 0.0
        for obs in observers:
            direct  = obs.attention_state.get(node_id, 0.0)
            basis   = clamp(sum(node.categories.get(c, 0.0) * w for c, w in obs.measurement_basis.items()) / max(len(obs.measurement_basis), 1))
            focus   = clamp((direct + basis) / 2.0)
            sal     = obs.coupling_strength * node.observer_sensitivity * focus
            total_salience += sal
            action     = obs.action_capacity.get(node_id, 0.0)
            intention  = obs.intention_state.get(node_id, 0.0)
            total_action += obs.coupling_strength * action * intention * params.alpha
        node.indeterminacy      = clamp(node.indeterminacy      - total_salience * 0.05 * dt)
        # v0.5: coherence boosted by attention but capped by disruption/status ceiling
        coherence_boost = total_salience * avg_coh * 0.03 * dt
        ceiling = coherence_ceiling(node)
        node.temporal_coherence = min(ceiling, clamp(node.temporal_coherence + coherence_boost))
        if total_action != 0.0:
            node.probability = clamp(node.probability + total_action * dt)
    normalize_branch_groups(world)
    if graph.nodes:
        ns = list(graph.nodes.values())
        world.temporal_state.temporal_coherence    = sum(n.temporal_coherence for n in ns) / len(ns)
        world.temporal_state.indeterminacy_density = sum(n.indeterminacy       for n in ns) / len(ns)
        world.temporal_state.temporal_entropy      = sum(n.temporal_entropy    for n in ns) / len(ns)
    if observers:
        world.temporal_state.observer_coupling = sum(o.coupling_strength for o in observers) / len(observers)
    return world


def compute_branch_potential_details(world):
    graph = world.event_graph
    groups: Dict[str, List[EventNode]] = {}
    for node in graph.nodes.values():
        if node.branch_group is not None:
            groups.setdefault(node.branch_group, []).append(node)
    group_scores = {}
    for gname, gnodes in groups.items():
        if len(gnodes) < 2: continue
        probs  = [max(n.probability, 1e-9) for n in gnodes]
        total  = sum(probs); normed = [p / total for p in probs]; n = len(normed)
        ent    = -sum(q * math.log(q) for q in normed) / math.log(n)
        ai = sum(nd.indeterminacy       for nd in gnodes) / len(gnodes)
        ae = sum(nd.temporal_energy     for nd in gnodes) / len(gnodes)
        ac = sum(nd.temporal_coherence  for nd in gnodes) / len(gnodes)
        score = ent * ai * ae * (1.0 - ac)
        group_scores[gname] = {
            "branch_score": round(clamp(score), 4), "entropy": round(ent, 4),
            "avg_indeterminacy": round(ai, 4), "avg_energy": round(ae, 4), "avg_coherence": round(ac, 4),
            "normalized_probabilities": {(nd.branch_label or nd.id): round(q, 4) for nd, q in zip(gnodes, normed)}
        }
    global_score = sum(v["branch_score"] for v in group_scores.values()) / len(group_scores) if group_scores else 0.0
    return {"global_branch_potential": round(clamp(global_score), 4), "groups": group_scores}


def compute_instability_score(world, params):
    graph = world.event_graph; ts = world.temporal_state
    conflict_score = 0.0
    for edge in graph.edges:
        if edge.relation_type in ("inhibitory", "exclusive"):
            s = graph.nodes[edge.source_id]; t = graph.nodes[edge.target_id]
            conflict_score += s.probability * t.probability * edge.causal_weight
    if graph.edges: conflict_score /= len(graph.edges)
    if conflict_score == 0.0:
        probs = [n.probability for n in graph.nodes.values()]
        mean_p = sum(probs) / max(len(probs), 1)
        conflict_score = sum((p - mean_p) ** 2 for p in probs) / max(len(probs), 1)
    ws = params.w_temporal_energy + params.w_indeterminacy + params.w_causal_conflict + params.w_entropy + params.w_coherence
    gi = clamp((params.w_temporal_energy * ts.temporal_flux + params.w_indeterminacy * ts.indeterminacy_density +
                params.w_causal_conflict * conflict_score + params.w_entropy * ts.temporal_entropy -
                params.w_coherence * ts.temporal_coherence) / ws)
    node_scores = {}
    for nid, node in graph.nodes.items():
        lc = 0.0; le = graph.incoming_edges(nid) + graph.outgoing_edges(nid)
        for edge in le:
            if edge.relation_type in ("inhibitory", "exclusive"):
                oid = edge.target_id if edge.source_id == nid else edge.source_id
                lc += node.probability * graph.nodes[oid].probability * edge.causal_weight
        if le: lc /= len(le)
        node_scores[nid] = clamp((params.w_temporal_energy * node.temporal_energy + params.w_indeterminacy * node.indeterminacy +
                                   params.w_causal_conflict * lc + params.w_entropy * node.temporal_entropy -
                                   params.w_coherence * node.temporal_coherence) / ws)
    bd = compute_branch_potential_details(world)
    world.temporal_state.branch_potential = bd["global_branch_potential"]
    probs_all = [n.probability for n in graph.nodes.values()]
    lambda_t = (max(probs_all) - min(probs_all)) * ts.causal_curvature if probs_all else 0.0
    stability = "CRITICAL" if gi >= 0.70 else "UNSTABLE" if gi >= params.instability_threshold else "STABLE"
    return {"global_instability": round(gi, 4), "causal_conflict": round(conflict_score, 4),
            "lambda_temporal": round(lambda_t, 4), "field_stability": stability,
            "node_scores": {k: round(v, 4) for k, v in node_scores.items()},
            "branch_details": bd}


def simulation_step(world, dt, params):
    world = apply_pending_evidence(world)
    world = compute_temporal_energy_sources(world, dt, params)
    world = propagate_temporal_flux(world, dt, params)
    world = apply_observer_effect(world, dt, params)
    scores = compute_instability_score(world, params)
    return world, scores


# ============================================================
# v0.4 TEST: Seeded from candidate events (simulates what live pipeline produces)
# ============================================================


# ============================================================
# v0.8: Two-Pass Seeding — Recompute after graph is built
# ============================================================

def recompute_coherence_and_energy(
    graph: 'EventGraph',
    observer_measurement_basis: Dict[str, float],
    seeded_causal_support_map: Dict[str, float]
) -> 'EventGraph':
    """
    Pass 3 & 4 of the seeding pipeline.

    After the graph is built and edges are in place, recompute causal_support
    from the actual graph structure, then re-derive temporal_coherence and
    temporal_energy using the updated value.

    This replaces the provisional causal_support=0.5 used in Pass 1.

    seeded_causal_support_map: dict of node_id -> original source_causal_evidence
    (the manually supplied causal_support from SeededEvent, as a placeholder
    for source-text extraction until the live extractor is connected)
    """
    for node_id, node in graph.nodes.items():
        # Step 1: recompute causal_support from graph
        source_ce = seeded_causal_support_map.get(node_id, 0.5)
        updated_cs = estimate_causal_support_from_graph(
            target_id=node_id,
            graph=graph,
            source_causal_evidence=source_ce
        )

        # Step 2: re-estimate temporal_coherence with updated causal_support
        record_coh = node.record_coherence
        new_coh = estimate_temporal_coherence(
            record_coherence=record_coh,
            causal_support=updated_cs,
            time_uncertainty=node.time_uncertainty,
            temporal_status=node.temporal_status,
            disruption_score=node.disruption_score,
            temporal_entropy=node.temporal_entropy
        )

        # Step 3: re-estimate temporal_energy with updated coherence
        obs_salience = estimate_observer_salience_from_categories(
            node.categories, observer_measurement_basis
        )
        branch_pressure = 1.0 if node.branch_group is not None else 0.3

        new_energy = estimate_initial_temporal_energy(
            indeterminacy=node.indeterminacy,
            temporal_entropy=node.temporal_entropy,
            downstream_impact=float(node.metadata.get("downstream_impact", 0.5)),
            observer_salience=obs_salience,
            branch_pressure=branch_pressure,
            temporal_coherence=new_coh,
            temporal_status=node.temporal_status
        )

        # Apply updates; track in auto_estimated
        node.temporal_coherence = new_coh
        node.temporal_energy    = new_energy
        node.auto_estimated["causal_support_graph"] = round(updated_cs, 3)
        node.auto_estimated["temporal_coherence_pass2"] = round(new_coh, 3)
        node.auto_estimated["temporal_energy_pass2"]    = round(new_energy, 3)
        node.metadata["causal_support"] = str(round(updated_cs, 3))

    return graph


def build_seeded_test_world() -> WorldTimelineState:
    """
    Demonstrates v0.4 seeding pipeline using pre-defined SeededEvents.
    In v0.4 with live data, these would come from:
      collect_candidate_items(topic) → AI extraction → SeededEvent list.
    """

    observer_basis = {
        "democratic_institutions": 0.9,
        "geopolitical_alliances":  0.8,
        "social_cohesion":         0.6,
        "economic_stability":      0.4
    }

    # Define candidate events using SeededEvent (no UOT fields — estimated below)
    candidates = [
        SeededEvent(
            id="e1", label="Serial cabinet firings (Bondi, Noem, Gabbard)",
            description="Three senior female cabinet members fired or forced out in sequence.",
            temporal_status="resolved", probability=0.95, time_uncertainty=0.05,
            categories={"democratic_institutions": 0.9, "social_cohesion": 0.6},
            sources=[SourceRef("Reuters report", "Reuters", stance="supporting", relevance=0.9),
                     SourceRef("WaPo analysis", "Washington Post", stance="supporting", relevance=0.85)],
            downstream_impact=0.75, disruption_score=0.7, novelty=0.6, causal_support=0.9,
            causal_candidates=["e2"],
            extraction_notes=ExtractionDiagnostics(
                downstream_impact_rationale="Forces all future nominees to signal total personal allegiance; affects DOJ, DHS, DNI downstream.",
                disruption_score_rationale="Breaks institutional independence norms; gendered pattern adds a second norm-breaking dimension.",
                novelty_rationale="Serial targeting of women in top posts is historically unusual; the pattern itself is the novelty.",
                probability_rationale="Already confirmed resolved; probability reflects factual certainty.",
                causal_rationale="Directly causes approval collapse by signaling instability to independent voters.",
                confidence=0.95
            )
        ),
        SeededEvent(
            id="e2", label="Approval rating collapse — 37%, independents at 34%",
            description="Sustained approval decline across all pollsters; stagflation concerns.",
            temporal_status="active", probability=0.92, time_uncertainty=0.1,
            categories={"democratic_institutions": 0.7, "economic_stability": 0.6, "social_cohesion": 0.7},
            sources=[SourceRef("Silver Bulletin", "Nate Silver", stance="supporting", relevance=0.95),
                     SourceRef("538 tracker", "ABC News", stance="supporting", relevance=0.9),
                     SourceRef("Rasmussen daily", "Rasmussen", stance="neutral", relevance=0.6)],
            downstream_impact=0.8, disruption_score=0.6, novelty=0.4, causal_support=0.85,
            causal_candidates=["e3a", "e3b", "e3c"]
        ),
        SeededEvent(
            id="e3a", label="Democrats flip House, fall short in Senate",
            description="D+7 generic ballot; Senate math requires net +4.",
            temporal_status="unresolved", probability=0.52, time_uncertainty=0.4,
            categories={"democratic_institutions": 0.9, "social_cohesion": 0.5},
            sources=[SourceRef("Cook Political", "Cook Political Report", stance="supporting", relevance=0.9),
                     SourceRef("Sabato Crystal Ball", "Sabato", stance="supporting", relevance=0.85)],
            downstream_impact=0.8, disruption_score=0.7, novelty=0.3, causal_support=0.7,
            branch_group="midterms_2026", branch_label="D_house_only",
            causal_candidates=["e5", "e7"]
        ),
        SeededEvent(
            id="e3b", label="Democratic sweep — House and Senate both flip",
            description="Would require near-perfect D performance in competitive Senate seats.",
            temporal_status="unresolved", probability=0.28, time_uncertainty=0.45,
            categories={"democratic_institutions": 0.95, "social_cohesion": 0.6},
            sources=[SourceRef("Polymarket", "Polymarket", stance="supporting", relevance=0.7),
                     SourceRef("Economist model", "The Economist", stance="neutral", relevance=0.8)],
            downstream_impact=0.9, disruption_score=0.8, novelty=0.4, causal_support=0.55,
            branch_group="midterms_2026", branch_label="D_sweep",
            causal_candidates=["e5"]
        ),
        SeededEvent(
            id="e3c", label="Republicans hold both chambers",
            description="Possible if gerrymandering/voter ID laws offset national headwinds.",
            temporal_status="unresolved", probability=0.20, time_uncertainty=0.4,
            categories={"democratic_institutions": 0.6, "social_cohesion": 0.4},
            sources=[SourceRef("RaceToTheWH", "RaceToTheWH", stance="contradicting", relevance=0.8)],
            downstream_impact=0.7, disruption_score=0.5, novelty=0.5, causal_support=0.3,
            branch_group="midterms_2026", branch_label="R_hold"
        ),
        SeededEvent(
            id="e4", label="NATO and post-WWII order further destabilized",
            description="Continued US pressure on NATO allies; Ukraine deal under duress.",
            temporal_status="unresolved", probability=0.72, time_uncertainty=0.35,
            categories={"geopolitical_alliances": 0.95, "economic_stability": 0.4},
            sources=[SourceRef("FT analysis", "Financial Times", stance="supporting", relevance=0.85),
                     SourceRef("Politico EU", "Politico Europe", stance="supporting", relevance=0.8)],
            downstream_impact=0.85, disruption_score=0.8, novelty=0.5, causal_support=0.75,
            causal_candidates=[]
        ),
        SeededEvent(
            id="e5", label="Post-midterm constitutional confrontation",
            description="Executive norm-breaking escalates if D House launches oversight.",
            temporal_status="unresolved", probability=0.55, time_uncertainty=0.5,
            categories={"democratic_institutions": 0.95, "geopolitical_alliances": 0.3},
            sources=[SourceRef("Legal scholars survey", "Yale Law", stance="supporting", relevance=0.7),
                     SourceRef("Brookings analysis", "Brookings", stance="neutral", relevance=0.75)],
            downstream_impact=0.9, disruption_score=0.9, novelty=0.6, causal_support=0.6,
            causal_candidates=["e6a"],
            extraction_notes=ExtractionDiagnostics(
                downstream_impact_rationale="Constitutional confrontation reshapes executive power for remainder of term; sets precedents for all future administrations.",
                disruption_score_rationale="Direct clash between branches over constitutional authority is systemic disruption by definition.",
                novelty_rationale="The combination of norm-breaking executive plus opposition House majority is historically unusual configuration.",
                probability_rationale="Conditional on D House flip (~68%); very high if that occurs given established confrontation pattern.",
                reconciliation_flags="Probability is conditional on e3a or e3b; should be modeled as dependent event.",
                confidence=0.70
            )
        ),
        SeededEvent(
            id="e6a", label="MAGA succession crisis — no viable heir",
            description="Personality-driven coalition lacks transferable charisma.",
            temporal_status="unresolved", probability=0.55, time_uncertainty=0.6,
            categories={"democratic_institutions": 0.6, "social_cohesion": 0.8},
            sources=[SourceRef("Axios deep dive", "Axios", stance="supporting", relevance=0.75)],
            downstream_impact=0.7, disruption_score=0.65, novelty=0.5, causal_support=0.55,
            branch_group="maga_succession", branch_label="crisis"
        ),
        SeededEvent(
            id="e6b", label="MAGA stabilizes behind viable heir (Vance)",
            description="Vance attempts to consolidate base; unclear if loyalty transfers.",
            temporal_status="unresolved", probability=0.28, time_uncertainty=0.6,
            categories={"democratic_institutions": 0.5, "social_cohesion": 0.5},
            sources=[SourceRef("Politico Vance profile", "Politico", stance="neutral", relevance=0.7)],
            downstream_impact=0.6, disruption_score=0.5, novelty=0.4, causal_support=0.45,
            branch_group="maga_succession", branch_label="stabilization"
        ),
        SeededEvent(
            id="e6c", label="MAGA fragments into competing factions",
            description="Multiple heirs compete; movement loses coherence.",
            temporal_status="unresolved", probability=0.17, time_uncertainty=0.65,
            categories={"democratic_institutions": 0.4, "social_cohesion": 0.9},
            sources=[SourceRef("Atlantic analysis", "The Atlantic", stance="neutral", relevance=0.7)],
            downstream_impact=0.75, disruption_score=0.8, novelty=0.6, causal_support=0.4,
            branch_group="maga_succession", branch_label="fragmentation"
        ),
        SeededEvent(
            id="e7", label="Democratic party internal fracture — left vs center",
            description="Squad wing vs. moderate wing tensions intensify post-midterms.",
            temporal_status="unresolved", probability=0.58, time_uncertainty=0.45,
            categories={"democratic_institutions": 0.5, "social_cohesion": 0.9},
            sources=[SourceRef("NYT analysis", "New York Times", stance="supporting", relevance=0.8),
                     SourceRef("Politico Playbook", "Politico", stance="neutral", relevance=0.7)],
            downstream_impact=0.65, disruption_score=0.6, novelty=0.3, causal_support=0.6,
            causal_candidates=["e6a", "e6b"]
        ),
    ]

    # ── Stage 0 + 0.5: query planning + SourcePackets ───────────────────────────
    queries = generate_search_queries("Trump presidency", observer_basis)
    print(f"[v0.12] Stage 0: {len(queries)} queries ({queries[0]['purpose']} + {len(queries)-1} targeted)")
    source_packets = web_search_sources("Trump presidency", observer_basis)
    print(f"[v0.12] {len(source_packets)} SourcePackets ready for Stage A:")
    for pkt in source_packets[:2]:  # show first two as sample
        print(f"  [{pkt.source_type}] {pkt.title} (cred={pkt.credibility:.2f}, rec={pkt.recency:.2f})")
        if pkt.branch_phrases:
            print(f"    Branch phrases: {pkt.branch_phrases[0][:70]}")
        if pkt.causal_phrases:
            print(f"    Causal phrases: {pkt.causal_phrases[0][:70]}")
    print()

    # ── PASS 1: estimate UOT fields with provisional causal_support ───────────
    estimated_nodes = [estimate_uot_fields(c, observer_basis) for c in candidates]

    # ── PASS 2: Stage C relational structure inference ──────────────────────
    # Stage C infers branch groups AND causal edges from the full event field.
    # In live mode: calls the AI with PROMPT_STAGE_C_STRUCTURE.
    # In stub mode: returns the pre-built structure for this scenario.
    print("[v0.12] Stage C: Relational structure inference (stub mode)...")
    structure = infer_branch_groups_and_causal_candidates_specialized(
        event_candidates=[{
            "id": c.id, "label": c.label, "probability": c.probability,
            "temporal_status": c.temporal_status,
            "downstream_impact": c.downstream_impact,
            "branch_hints": [], "causal_hints": []
        } for c in candidates],
        source_summaries="Trump presidency test scenario (stub)",
        topic="Trump presidency"
    )
    # Apply Stage C structure back to estimated_nodes
    estimated_nodes_dicts = [{**n} for n in estimated_nodes]
    updated_dicts, final_edges = apply_structure_to_event_candidates(
        estimated_nodes_dicts, structure
    )
    # Merge branch/label assignments back into estimated_nodes
    id_map = {n["id"]: n for n in updated_dicts}
    for node in estimated_nodes:
        updated = id_map.get(node["id"], {})
        if updated.get("branch_group"):
            node["branch_group"] = updated["branch_group"]
            node["branch_label"] = updated["branch_label"]

    n_bg = len(structure.get("branch_groups",[]))
    n_ed = len(structure.get("causal_edges",[]))
    print(f"         Stage C: {n_bg} branch groups, {n_ed} causal edges")

    # --- Print review summary (observer review step) ---
    print_review_summary(estimated_nodes, final_edges, seeded_events=candidates)

    # ── PASS 2: Build graph ─────────────────────────────────────────────────
    graph = build_graph_from_estimates(estimated_nodes, final_edges)

    # ── PASS 3 & 4: Recompute causal_support from graph, then coherence/energy ─
    # Build source_causal_evidence map from original candidates
    seeded_causal_support_map = {c.id: c.causal_support for c in candidates}
    graph = recompute_coherence_and_energy(graph, observer_basis, seeded_causal_support_map)
    print("\n[v0.8] Two-pass seeding complete: causal_support recomputed from graph structure.")

    # --- Observer ---
    observer = ObserverState(
        id="vince", label="Vince — analyst",
        measurement_basis=observer_basis,
        attention_state={"e5": 0.9, "e4": 0.85, "e6a": 0.8, "e1": 0.6},
        action_capacity={"e7": 0.1},
        coherence_level=0.8, coupling_strength=0.6
    )

    # --- Evidence events (observer's reads) ---
    evidence = [
        EvidenceEvent("ev1", "e3b", 0.75, confidence=0.8, source="Vince",
                      description="D sweep less likely than House-only; Senate math hard"),
        EvidenceEvent("ev2", "e5",  1.3,  confidence=0.7, source="Vince",
                      description="Constitutional confrontation more likely than base rate"),
        EvidenceEvent("ev3", "e3c", 0.4,  confidence=0.9, source="Polling + markets",
                      description="Independent approval at 34%; R hold very unlikely"),
        EvidenceEvent("ev4", "e6a", 1.2,  confidence=0.65, source="Vince",
                      description="MAGA without Trump lacks transferable charisma"),
    ]

    return WorldTimelineState(
        event_graph=graph, observers=[observer],
        evidence_events=evidence, temporal_state=TemporalState()
    )


# ============================================================
# v0.9: Live-Data Extraction Pipeline
# ============================================================

# Read from environment: export LIVE_MODE=true
LIVE_MODE = os.getenv("LIVE_MODE", "false").lower() in ("1", "true", "yes")

PROMPT_HOLISTIC_EXTRACTION = """
You are a UOT temporal extrapolation engine. Identify 5–10 candidate events
for a Unified Observer Theory timeline graph from the topic and sources.

Topic: {topic}
Observer measurement basis:
{observer_basis}
Source summaries:
{source_summaries}

For each event return:
  id, label, description, temporal_status (resolved|active|unresolved|counterfactual),
  probability, time_estimate (null or 0-3), time_uncertainty (0-1),
  categories dict (democratic_institutions, geopolitical_alliances, social_cohesion,
  economic_stability each 0-1), sources list, source_count, recency (0-1),
  probability_rationale.

Also return HINTS (not final structure — Stage C will finalize these):
  branch_hints: list of strings describing possible mutual exclusivity with other events.
    Example: "Mutually exclusive with Democratic sweep and Republican hold outcomes"
  causal_hints: list of strings describing possible causal relationships.
    Example: "Approval collapse increases probability of this outcome"

Do NOT set branch_group, branch_label, or causal_candidates — leave null/empty.
Do NOT estimate downstream_impact, disruption_score, or novelty yet.
Keep descriptions under 80 words. Keep probability_rationale under 30 words.
Aim for 6-8 events maximum to keep response concise.

Return format:
{{
  "events": [...],
  "follow_up_search_queries": [],
  "missing_evidence": []
}}
Return valid JSON only. No prose, no fences.
""".strip()

# Combined Stage B prompt: scores all three dimensions in one call.
# Replaces three separate prompts to reduce API calls from 6 to 4.
PROMPT_SCORE_COMBINED = """
For each candidate event below, estimate three scores simultaneously.
Return all three for every event in one response.

DOWNSTREAM IMPACT (0-1): how many significant later events, institutions,
actors, or branches this event is likely to affect.
  0.0-0.2 isolated; 0.3-0.4 one domain; 0.5-0.6 one major system;
  0.7-0.8 multiple systems; 0.9-1.0 reshapes whole graph.

DISRUPTION SCORE (0-1): how much the event destabilizes existing order,
norms, institutions, alliances, or system continuity.
  0.0-0.2 routine; 0.3-0.4 mild; 0.5-0.6 notable but functional;
  0.7-0.8 severe; 0.9-1.0 systemic rupture.

NOVELTY (0-1): how historically unusual, unprecedented, or low-precedent.
  0.0-0.2 routine; 0.3-0.4 minor unusual features; 0.5-0.6 meaningfully unusual;
  0.7-0.8 historically exceptional; 0.9-1.0 unprecedented.

Candidate events:
{events_summary}

Return JSON only:
{{"event_id": {{"downstream_impact": 0.0, "disruption_score": 0.0, "novelty": 0.0,
              "di_rationale": "...", "ds_rationale": "...", "nv_rationale": "..."}}}}
""".strip()


PROMPT_SCORE_DOWNSTREAM_IMPACT = """
For each event, estimate downstream_impact 0-1.
Definition: how many significant later events, institutions, actors, or branches
this event is likely to affect.
Rubric: 0.0-0.2 isolated; 0.3-0.4 one domain; 0.5-0.6 one major system;
0.7-0.8 multiple systems; 0.9-1.0 reshapes whole graph.
Score only downstream consequence scale — not surprise or disruption.
Events: {events_summary}
Return JSON: {{"event_id": {{"score": 0.0, "rationale": "..."}}}}
""".strip()

PROMPT_SCORE_DISRUPTION = """
For each event, estimate disruption_score 0-1.
Definition: how much the event destabilizes existing order, norms, institutions,
alliances, expectations, or system continuity.
Rubric: 0.0-0.2 routine; 0.3-0.4 mild; 0.5-0.6 notable but functional;
0.7-0.8 severe; 0.9-1.0 systemic rupture.
Score only destabilization — not downstream scale.
Events: {events_summary}
Return JSON: {{"event_id": {{"score": 0.0, "rationale": "..."}}}}
""".strip()

PROMPT_SCORE_NOVELTY = """
For each event, estimate novelty 0-1.
Definition: how historically unusual, surprising, or low-precedent the event is.
Rubric: 0.0-0.2 routine; 0.3-0.4 minor unusual features; 0.5-0.6 meaningfully unusual;
0.7-0.8 historically exceptional; 0.9-1.0 unprecedented.
Score only historical unusualness — not importance or disruption.
Events: {events_summary}
Return JSON: {{"event_id": {{"score": 0.0, "rationale": "..."}}}}
""".strip()

PROMPT_RECONCILE = """
Review these candidate events and scores for internal consistency.
Flag: downstream/disruption/novelty conflation; probability vs source agreement conflicts;
non-mutually-exclusive branch groups; temporal order violations; events needing merge/split.
Events: {events_with_scores}
Return JSON: {{"event_id": ["flag..."]}}. Empty dict if no flags.
""".strip()


PROMPT_STAGE_C_STRUCTURE = """
Given the full candidate event set below (with scores and Stage A hints),
infer the relational structure of the UOT timeline graph.

Candidate events:
{events_with_hints}

Topic: {topic}

Tasks:
1. BRANCH GROUPS: mutually exclusive alternative outcomes of the same question.
   Only group events that truly cannot co-occur.
2. CAUSAL EDGES: directed relationships.
   relation_type: causal | reinforcing | inhibitory | exclusive | enabling | feedback
   causal_weight 0-1, uncertainty 0-1, feedback_strength 0-1.

Return JSON only:
{{
  "branch_groups": [
    {{"branch_group": "key", "resolution_question": "...",
      "members": [{{"event_id": "e1", "branch_label": "label"}}],
      "mutual_exclusivity_confidence": 0.9, "rationale": "..."}}
  ],
  "causal_edges": [
    {{"source_id": "e1", "target_id": "e2", "relation_type": "causal",
      "causal_weight": 0.6, "uncertainty": 0.3, "feedback_strength": 0.0,
      "rationale": "..."}}
  ]
}}
""".strip()


# ============================================================
# v0.11: Source Compression Layer
# ============================================================
# Sits before Stage A. Transforms raw fetched text into
# structured SourcePackets that preserve causal and branch
# language while filtering noise.
# ============================================================

PROMPT_COMPRESS_SOURCE = """
You are a source compression engine for a temporal extrapolation system.
Given a news article or source text, extract structured observational content.

Source metadata:
Title: {title}
Publisher: {publisher}
Date: {date}
URL: {url}

Article text (may be truncated):
{text}

Extract the following fields:
- summary: 2-3 sentence factual summary of the main claims
- key_claims: list of 3-6 specific factual claims (not opinions)
- causal_phrases: list of phrases that indicate causation, mechanism, or consequence
  (look for: "led to", "because", "due to", "triggered", "resulted in", "caused",
   "drove", "as a result", "following", "after", "in response to")
- branch_phrases: list of phrases that indicate conditional futures or alternatives
  (look for: "could", "might", "if X then Y", "unless", "depends on", "may",
   "either...or", "two scenarios", "possible outcomes", "would likely")
- evidence_snippets: list of 1-3 short direct quotes or specific facts (under 30 words each)
  that carry the most evidential weight
- source_type: article | polling | legal_analysis | market | official | academic
- credibility: 0-1 estimate of source reliability (0.9 for major wire service,
  0.7 for quality press, 0.5 for opinion, 0.3 for partisan)
- recency: 0-1 (1.0 = today, 0.8 = this week, 0.5 = this month, 0.2 = this year)

Return JSON only. No prose, no fences.
{{
  "summary": "...",
  "key_claims": ["..."],
  "causal_phrases": ["..."],
  "branch_phrases": ["..."],
  "evidence_snippets": ["..."],
  "source_type": "article",
  "credibility": 0.7,
  "recency": 0.8
}}
""".strip()


def _as_dict(value, fallback=None):
    """Return value if it is a dict, otherwise return fallback (default {})."""
    return value if isinstance(value, dict) else (fallback if fallback is not None else {})

def _as_list(value):
    """Return value as a clean list of strings regardless of input shape."""
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []

def _safe_float(value, default=0.5):
    """Return float(value) or default if conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compress_source_to_packet(
    raw_text: str,
    metadata: dict
) -> SourcePacket:
    """
    Transforms a raw fetched source into a structured SourcePacket.

    In live mode: calls the AI with PROMPT_COMPRESS_SOURCE.
    In stub mode: builds a minimal packet from metadata.

    This is the key v0.11 addition: Stage A no longer receives
    raw text — it receives this structured packet.
    """
    meta      = _as_dict(metadata)  # always safe to call .get() on meta
    title     = meta.get("title", "")
    publisher = meta.get("publisher", "")
    date      = meta.get("date", "")
    url       = meta.get("url", "")

    if not LIVE_MODE:
        # Stub: build a minimal but structurally correct packet
        return SourcePacket(
            title=title, publisher=publisher, url=url, date=date,
            source_type=meta.get("source_type", "article"),
            credibility=float(meta.get("credibility", 0.7)),
            recency=float(meta.get("recency", 0.7)),
            summary=f"[stub] {title}",
            key_claims=[f"[stub] Key claim from {publisher}"],
            causal_phrases=meta.get("causal_phrases", []),
            branch_phrases=meta.get("branch_phrases", []),
            evidence_snippets=[],
            raw_text_ref=meta.get("cache_key", ""),
            raw_text_chars=len(raw_text)
        )

    # Truncate text to ~3000 chars for compression prompt
    text_truncated = raw_text[:3000] + ("..." if len(raw_text) > 3000 else "")

    result = call_anthropic_api(
        "You are a source compression engine. Return only valid JSON.",
        PROMPT_COMPRESS_SOURCE.format(
            title=title, publisher=publisher, date=date, url=url,
            text=text_truncated
        )
    )

    # Guard: normalize result before any .get() calls
    if not isinstance(result, dict):
        result = {
            "summary":           str(result)[:800] if result else "",
            "key_claims":        [], "causal_phrases":    [],
            "branch_phrases":    [], "evidence_snippets": [],
            "source_type":       meta.get("source_type", "article") if isinstance(metadata, dict) else "article",
            "credibility":       0.7,
            "recency":           0.7,
        }
    meta = _as_dict(metadata)
    return SourcePacket(
        title=title, publisher=publisher, url=url, date=date,
        source_type=str(result.get("source_type", "article")),
        credibility=clamp(_safe_float(result.get("credibility"), 0.7)),
        recency    =clamp(_safe_float(result.get("recency"),     0.7)),
        summary    =str(result.get("summary", "")),
        key_claims      =_as_list(result.get("key_claims",        [])),
        causal_phrases  =_as_list(result.get("causal_phrases",    [])),
        branch_phrases  =_as_list(result.get("branch_phrases",    [])),
        evidence_snippets=_as_list(result.get("evidence_snippets", [])),
        raw_text_ref    =meta.get("cache_key", url),
        raw_text_chars  =len(raw_text or "")
    )


def format_source_packets_for_stage_a(packets: List[SourcePacket]) -> str:
    """
    Formats SourcePackets into the structured string Stage A receives.

    Preserves: summary, key claims, causal language, branch language,
    evidence snippets, credibility, recency.

    Excludes: raw text, cache keys, internal metadata.

    This is the observational record the Stage A prompt reads.
    """
    lines = []
    for i, pkt in enumerate(packets, 1):
        lines.append(f"SOURCE {i}: {pkt.title}")
        lines.append(f"  Publisher: {pkt.publisher}  |  Date: {pkt.date}  |  "
                     f"Type: {pkt.source_type}  |  "
                     f"Credibility: {pkt.credibility:.2f}  |  Recency: {pkt.recency:.2f}")
        if pkt.summary:
            lines.append(f"  Summary: {pkt.summary}")
        if pkt.key_claims:
            lines.append("  Key claims:")
            for claim in pkt.key_claims[:4]:
                lines.append(f"    - {claim}")
        if pkt.causal_phrases:
            lines.append("  Causal language:")
            for phrase in pkt.causal_phrases[:3]:
                lines.append(f"    ↳ {phrase}")
        if pkt.branch_phrases:
            lines.append("  Branch/conditional language:")
            for phrase in pkt.branch_phrases[:3]:
                lines.append(f"    ⇒ {phrase}")
        if pkt.evidence_snippets:
            lines.append("  Evidence snippets:")
            for snippet in pkt.evidence_snippets[:2]:
                lines.append("    [" + snippet[:60] + "]")
        lines.append("")
    return "\n".join(lines)


def diversify_and_rank_sources(packets: List[SourcePacket]) -> List[SourcePacket]:
    """
    Two-pass deduplication and ranking (v0.12 fix).

    The previous single-pass used a mutable seen_types set inside the sort
    key, making the diversity bonus order-dependent and non-deterministic.

    Pass 1: Deduplicate by URL; score each packet by credibility + recency.
    Pass 2: Type-aware interleaving — round-robin by source_type first,
            then fill remaining slots from the sorted remainder.
    """
    # Pass 1: deduplicate + score
    seen_urls: set = set()
    scored: List[tuple] = []
    for pkt in packets:
        key = pkt.url or pkt.title
        if key not in seen_urls:
            seen_urls.add(key)
            base_score = pkt.credibility * 0.5 + pkt.recency * 0.4
            scored.append((base_score, pkt))
    scored.sort(key=lambda x: x[0], reverse=True)

    # Pass 2: group by source_type, round-robin interleave
    by_type: Dict[str, List] = {}
    for score_val, pkt in scored:
        by_type.setdefault(pkt.source_type, []).append((score_val, pkt))

    result: List[SourcePacket] = []
    type_queues = list(by_type.values())

    # One best from each type first
    for queue in type_queues:
        if len(result) >= 8:
            break
        if queue:
            result.append(queue.pop(0)[1])

    # Fill remaining from best-scoring remainder
    remainder = sorted(
        [item for queue in type_queues for item in queue],
        key=lambda x: x[0], reverse=True
    )
    for _, pkt in remainder:
        if len(result) >= 8:
            break
        result.append(pkt)

    return result


def expand_event_with_raw_sources(
    event_id: str,
    raw_text_refs: List[str],
    raw_text_cache: dict
) -> dict:
    """
    Fallback: fetches full text for a specific weak or disputed event.
    Used when Stage A confidence is low or Stage D flags an event.

    In live deployment:
      - raw_text_cache maps raw_text_ref keys to full article text
      - Runs a targeted extraction prompt for just this event
      - Returns updated scores and rationale

    This gives full-text power for audit/recovery without flooding Stage A.
    """
    if not LIVE_MODE:
        return {"event_id": event_id, "status": "stub — no expansion in LIVE_MODE=False"}

    full_texts = [raw_text_cache.get(ref, "") for ref in raw_text_refs if ref]
    combined   = "\n\n---\n\n".join(full_texts)[:6000]  # 6000 char limit

    result = call_anthropic_api(
        "You are a UOT temporal extrapolation engine. Return only valid JSON.",
        f"""Re-analyze event '{event_id}' using full source text.

Full source text:
{combined}

Return updated estimates for this event only:
{{
  "event_id": "{event_id}",
  "probability": 0.0,
  "downstream_impact": 0.0,
  "disruption_score": 0.0,
  "novelty": 0.0,
  "causal_support": 0.0,
  "updated_rationale": "...",
  "confidence": 0.0
}}"""
    )
    return result


# ============================================================
# v0.12: Hybrid Query Fan-Out System
# ============================================================

def generate_search_queries(topic: str, observer_basis: dict) -> List[dict]:
    """
    Stage 0: Query planning — hybrid fan-out.
    One broad field scan + narrow measurement-basis projections.
    Stage A observes the result; Stage A does NOT plan the initial queries.
    """
    queries = [
        {"query": f"{topic} latest developments 2026",
         "purpose": "field_scan", "source_type": "article", "max_results": 5},
        {"query": f"{topic} approval rating polling latest 2026",
         "purpose": "public_opinion", "source_type": "polling", "max_results": 3},
        {"query": f"{topic} election forecast midterms 2026",
         "purpose": "branch_detection", "source_type": "forecast", "max_results": 3},
    ]
    category_queries = {
        "democratic_institutions":
            {"query": f"{topic} constitutional legal institutional analysis 2026",
             "purpose": "democratic_institutions", "source_type": "legal_analysis"},
        "geopolitical_alliances":
            {"query": f"{topic} NATO allies foreign policy geopolitical 2026",
             "purpose": "geopolitical_alliances", "source_type": "article"},
        "social_cohesion":
            {"query": f"{topic} party factions succession political movement 2026",
             "purpose": "succession_branching", "source_type": "article"},
        "economic_stability":
            {"query": f"{topic} economy tariffs inflation economic impact 2026",
             "purpose": "economic_stability", "source_type": "market"},
    }
    for cat, weight in observer_basis.items():
        if weight >= 0.6 and cat in category_queries:
            queries.append(category_queries[cat])
    return queries


def run_searches(topic: str, observer_basis: dict) -> List[SearchResult]:
    """
    Stage 0.5 (first pass): Executes hybrid query fan-out.
    Deduplicates results by URL across all queries.
    In stub mode: returns pre-built SearchResults.
    """
    if not LIVE_MODE:
        return [
            SearchResult("Trump presidency latest developments 2026", "field_scan",
                "Trump approval rating hits record low", "Silver Bulletin",
                "https://natesilver.net/approval", "2026-06-01", "", "polling", 1),
            SearchResult("Trump presidency approval rating polling latest 2026", "public_opinion",
                "Trump net approval -19.1", "Strength In Numbers",
                "https://gelliottmorris.com/poll", "2026-05-26", "", "polling", 1),
            SearchResult("Trump presidency election forecast midterms 2026", "branch_detection",
                "Democrats favored to flip House", "Cook Political Report",
                "https://cookpolitical.com", "2026-05-28", "", "forecast", 1),
            SearchResult("Trump presidency constitutional legal analysis 2026", "democratic_institutions",
                "Constitutional guardrails under strain", "Brookings",
                "https://brookings.edu/analysis", "2026-05-10", "", "legal_analysis", 1),
            SearchResult("Trump presidency NATO allies foreign policy 2026", "geopolitical_alliances",
                "NATO allies accelerate independent defense", "Financial Times",
                "https://ft.com/nato", "2026-05-15", "", "article", 1),
            SearchResult("Trump presidency party factions succession 2026", "succession_branching",
                "MAGA succession: who comes after Trump?", "Axios",
                "https://axios.com/maga-succession", "2026-05-20", "", "article", 1),
        ]
    # Live: execute queries with deduplication
    seen_urls: set = set()
    results: List[SearchResult] = []
    for q in generate_search_queries(topic, observer_basis):
        # call search API here
        pass
    return results


def run_followup_searches(follow_up_queries: List[str]) -> List[SearchResult]:
    """
    Stage 0.6: Adaptive follow-up from Stage A evidence gap requests.
    In stub mode: returns empty (stub Stage A returns no follow-ups).
    """
    if not LIVE_MODE or not follow_up_queries:
        return []
    # Live: run up to 3 follow-up queries
    return []


def fetch_top_results(search_results: List[SearchResult], max_docs: int = 8) -> List[RawDocument]:
    """
    Fetches full text for top search results. Returns RawDocuments with cache keys.
    """
    if not LIVE_MODE:
        import hashlib
        docs = []
        for sr in search_results[:max_docs]:
            ck = hashlib.md5(sr.url.encode()).hexdigest()[:12]
            docs.append(RawDocument(
                title=sr.title, publisher=sr.publisher, url=sr.url, date=sr.date,
                text=f"[stub] Full text for: {sr.title}",
                metadata={
                    "title": sr.title, "publisher": sr.publisher, "url": sr.url,
                    "date": sr.date, "source_type": sr.source_type_hint,
                    "credibility": 0.85 if sr.source_type_hint in ("polling","forecast") else 0.75,
                    "recency": 0.90, "purpose": sr.purpose, "cache_key": ck
                },
                cache_key=ck
            ))
        return docs
    return []



def web_search_sources(topic: str, observer_basis: Optional[dict] = None, follow_up_queries: Optional[List[str]] = None) -> List[SourcePacket]:
    """
    Three-layer source pipeline:
      Layer 1: Run searches and fetch top results
      Layer 2: Compress each raw document into a SourcePacket
      Layer 3: Diversify and rank packets

    In stub mode: returns pre-built SourcePackets for the test scenario.
    In live mode: implement run_searches() + fetch_top_results() here.
    """
    if not LIVE_MODE:
        # Stub packets for the Trump presidency scenario
        # In live mode, these come from actual web search + compression
        return [
            SourcePacket(
                title="Trump approval rating hits record low", publisher="Silver Bulletin",
                date="2026-06-01", source_type="polling", credibility=0.88, recency=0.95,
                summary="Trump's net approval sits at -19.1, worse than Biden at same point. Independent approval at 34%.",
                key_claims=["37% overall approval", "34% independent approval",
                            "48% strongly disapprove", "Worse than any prior Trump reading"],
                causal_phrases=["driven by tariff-driven inflation", "eroded by stagflation concerns",
                                "accelerated after government shutdown"],
                branch_phrases=["could reach wave election territory if independents stay below 36%"],
                evidence_snippets=["net approval rating -19.1, less popular than Biden at this point"]
            ),
            SourcePacket(
                title="Democrats favored to flip House in 2026", publisher="Cook Political Report",
                date="2026-05-28", source_type="polling", credibility=0.90, recency=0.92,
                summary="Democrats are clear favorites to flip the House; Senate math is harder, requiring net +4.",
                key_claims=["73% market probability of D House win", "40% chance of Democratic sweep",
                            "Republicans need to hold 23 of 35 Senate seats up",
                            "D+7 generic ballot"],
                causal_phrases=["approval collapse leads to wave conditions",
                                "special election results reinforce Democratic momentum"],
                branch_phrases=["Democrats could win House but fall short in Senate",
                                "if Republicans redraw maps in Florida and Tennessee",
                                "Democratic sweep possible if Senate races break simultaneously"],
                evidence_snippets=["Economist model gives Democrats heavy House favorites in 25,001 simulations"]
            ),
            SourcePacket(
                title="Trump fires AG Bondi after intelligence leak", publisher="Reuters",
                date="2026-04-02", source_type="official", credibility=0.92, recency=0.80,
                summary="Trump fired Pam Bondi following reports she leaked sensitive intelligence. Third senior firing in sequence after Noem and Gabbard.",
                key_claims=["Bondi fired April 2026", "Noem and Gabbard previously forced out",
                            "Pattern: women in senior positions targeted first"],
                causal_phrases=["led to further cabinet instability", "resulted in growing perception of loyalty-only governance"],
                branch_phrases=["may signal further senior departures"],
                evidence_snippets=["Third consecutive female cabinet member to exit under pressure"]
            ),
            SourcePacket(
                title="NATO allies increase defense spending as US commitment wavers",
                publisher="Financial Times", date="2026-05-15",
                source_type="article", credibility=0.88, recency=0.85,
                summary="European NATO members accelerating independent defense capacity as Trump signals potential US withdrawal.",
                key_claims=["Germany increases defense budget to 3% GDP", "EU defense coordination accelerating",
                            "Trump called NATO a 'paper tiger'"],
                causal_phrases=["triggered by US withdrawal signals", "led to independent EU defense planning",
                                "driven by unreliable US commitment signals"],
                branch_phrases=["could trigger formal restructuring of alliance",
                                "if US withdraws, European defense bloc would likely emerge"],
                evidence_snippets=["EU defense spending up 18% year-over-year across member states"]
            ),
            SourcePacket(
                title="MAGA succession: who comes after Trump?", publisher="Axios",
                date="2026-05-20", source_type="article", credibility=0.82, recency=0.86,
                summary="Analysis of whether Vance, DeSantis, or another figure can consolidate the MAGA coalition post-Trump.",
                key_claims=["No single heir has consolidated base support", "Vance leads in VP positioning",
                            "Trump coalition is personality-dependent, not ideology-dependent"],
                causal_phrases=["movement cohesion depends on Trump remaining central figure"],
                branch_phrases=["could fracture into competing factions", "might stabilize behind Vance",
                                "three possible succession paths: crisis, consolidation, fragmentation"],
                evidence_snippets=["Historical precedent: personality-driven movements rarely survive intact after founder exits"]
            )
        ]

    # Live mode: use uot_live_search module for Stage 0 + Stage 0.5.
    # Stage 0:   run_searches_live()      → Tavily or Brave Search API
    # Stage 0.5: fetch_top_results_live() → direct HTTP fetch + text extraction
    # compress_source_to_packet() and Stage A-D remain Anthropic-based.
    #
    # Setup: export TAVILY_API_KEY='tvly-xxxxxxxxxxxx'
    # Test:  python uot_live_search.py "your topic"
    if observer_basis is None: observer_basis = {}
    try:
        from uot_live_search import live_source_pipeline
    except ImportError:
        raise RuntimeError(
            "uot_live_search.py not found. "
            "Place it in the same directory as this engine file."
        )
    raw_docs = live_source_pipeline(topic, observer_basis, follow_up_queries)
    packets  = [compress_source_to_packet(doc.text, doc.metadata) for doc in raw_docs]
    return diversify_and_rank_sources(packets)


def _repair_truncated_json(text: str):
    """
    Salvage a JSON response truncated by max_tokens.
    Finds the last complete JSON object and closes any open structures.
    Returns a parsed dict/list or raises ValueError.
    """
    import json

    # First: strip markdown fences if present
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    # Try direct parse first
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass

    # Find the last position of a closing brace/bracket
    # by scanning and tracking open depth
    depth_brace = 0
    depth_bracket = 0
    in_string = False
    escape_next = False
    last_complete = -1   # index of last char where depth returned to 0

    for i, ch in enumerate(t):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            depth_brace   += (ch == '{')
            depth_bracket += (ch == '[')
        elif ch == '}':
            depth_brace -= 1
            if depth_brace == 0 and depth_bracket <= 1:
                last_complete = i
        elif ch == ']':
            depth_bracket -= 1
            if depth_bracket == 0 and depth_brace == 0:
                last_complete = i

    if last_complete < 0:
        raise ValueError("No complete JSON structure found")

    # Truncate to last complete object and close open structures
    truncated = t[:last_complete + 1]

    # Recount what's still open after truncation
    db = 0; dbl = 0; ins = False; esc = False
    for ch in truncated:
        if esc: esc = False; continue
        if ch == '\\' and ins: esc = True; continue
        if ch == '"': ins = not ins; continue
        if ins: continue
        if ch == '{': db += 1
        elif ch == '}': db -= 1
        elif ch == '[': dbl += 1
        elif ch == ']': dbl -= 1

    closing = (']' * max(0, dbl)) + ('}' * max(0, db))
    repaired = truncated + closing

    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not repair truncated JSON: {e}")


def call_anthropic_api(system_prompt: str, user_message: str,
                       model: str = "claude-sonnet-4-6",
                       max_tokens: int = 4000,
                       timeout: int = 90) -> dict:
    """
    Calls Anthropic API and returns parsed JSON.

    Requires ANTHROPIC_API_KEY environment variable.
    Set before running: export ANTHROPIC_API_KEY="sk-ant-..."

    Use model="claude-haiku-4-5-20251001" for faster/cheaper scoring calls.
    """
    import json, urllib.request
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. "
            "Set it with: export ANTHROPIC_API_KEY='sk-ant-...'"
        )
    payload = json.dumps({
        "model": model, "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}]
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=payload,
        headers={
            "Content-Type":    "application/json",
            "x-api-key":       api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        raise RuntimeError(f"Anthropic API HTTP {e.code}: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Anthropic API network error: {e.reason}")
    # Guard each content block — some may be strings, not dicts
    content = data.get("content", [])
    parts = []
    for block in (content if isinstance(content, list) else []):
        if isinstance(block, dict):
            parts.append(block.get("text", ""))
        elif isinstance(block, str):
            parts.append(block)
    text = "".join(parts).strip()
    if text.startswith("```"):
        text = text.split("\n",1)[-1].rsplit("```",1)[0].strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Response may have been truncated by max_tokens — attempt repair
        result = _repair_truncated_json(text)
    # Ensure we never return a bare string — callers always expect dict or list
    if isinstance(result, str):
        raise ValueError(f"API returned a JSON string instead of object: {result[:80]}")
    return result


def extract_candidate_events_holistic(topic, source_summaries, observer_basis):
    """
    Stage A: holistic extraction + evidence gap detection.
    Returns (event_dicts, follow_up_queries, missing_evidence).
    """
    if not LIVE_MODE:
        return [], [], []
    basis_str = "\n".join(f"  {k}: {v}" for k,v in observer_basis.items())
    result = call_anthropic_api(
        "You are a UOT temporal extrapolation engine. Return only valid JSON.",
        PROMPT_HOLISTIC_EXTRACTION.format(
            topic=topic, observer_basis=basis_str, source_summaries=source_summaries),
        model="claude-sonnet-4-6",
        max_tokens=8000,   # Stage A needs room for 8-10 events with all fields
        timeout=90
    )
    if isinstance(result, list):
        return result, [], []
    return result.get("events", []), result.get("follow_up_search_queries", []), result.get("missing_evidence", [])


def score_event_fields_specialized(event_candidates, source_summaries):
    """
    Stage B: single combined scoring call (replaces three separate calls).
    Uses claude-haiku for speed — 3 calls reduced to 1.
    """
    if not LIVE_MODE:
        return event_candidates
    summary = "\n".join(
        f"- {e['id']}: {e['label']} | {e.get('description','')[:80]}"
        for e in event_candidates
    )
    try:
        scores = call_anthropic_api(
            "Return only valid JSON. No prose, no markdown fences.",
            PROMPT_SCORE_COMBINED.format(events_summary=summary),
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            timeout=90
        )
    except Exception as e:
        # If combined scoring fails, use neutral defaults and continue
        scores = {}

    for ev in event_candidates:
        eid  = ev["id"] if isinstance(ev, dict) else str(ev)
        if not isinstance(ev, dict):
            continue
        raw  = scores.get(eid, {}) if isinstance(scores, dict) else {}
        data = raw if isinstance(raw, dict) else {}
        ev["downstream_impact"] = clamp(float(data.get("downstream_impact", 0.5)))
        ev["disruption_score"]  = clamp(float(data.get("disruption_score",  0.5)))
        ev["novelty"]           = clamp(float(data.get("novelty",           0.5)))
        ev["extraction_notes"]  = {
            "downstream_impact_rationale": data.get("di_rationale", ""),
            "disruption_score_rationale":  data.get("ds_rationale", ""),
            "novelty_rationale":           data.get("nv_rationale", ""),
            "probability_rationale":       ev.get("probability_rationale", ""),
            "confidence": 0.7
        }
    return [ev for ev in event_candidates if isinstance(ev, dict)]


def reconcile_scores_for_consistency(event_candidates):
    """Stage D: consistency check across the whole event field."""
    if not LIVE_MODE:
        return event_candidates
    scored_summary = "\n".join(
        f"- {e['id']}: P={e.get('probability',0.5):.2f} "
        f"DI={e.get('downstream_impact',0.5):.2f} "
        f"DS={e.get('disruption_score',0.5):.2f} "
        f"NV={e.get('novelty',0.5):.2f} status={e.get('temporal_status','?')}"
        for e in event_candidates
    )
    flags = call_anthropic_api("Return only valid JSON.",
        PROMPT_RECONCILE.format(events_with_scores=scored_summary))
    for ev in event_candidates:
        if not isinstance(ev, dict): continue
        flag_data = flags.get(ev["id"], {}) if isinstance(flags, dict) else {}
        if not isinstance(flag_data, dict): flag_data = {}
        if ev["id"] in flags and flag_data:
            notes = ev.get("extraction_notes") or {}
            notes["reconciliation_flags"] = "; ".join(flags[ev["id"]])
            ev["extraction_notes"] = notes
    return event_candidates


def build_seeded_events_from_dicts(event_dicts):
    """Converts raw extractor output dicts into SeededEvent objects."""
    seeded = []
    for d in (x for x in (event_dicts or []) if isinstance(x, dict)):
        nd = d.get("extraction_notes") or {}
        diag = ExtractionDiagnostics(
            downstream_impact_rationale = nd.get("downstream_impact_rationale",""),
            disruption_score_rationale  = nd.get("disruption_score_rationale",""),
            novelty_rationale           = nd.get("novelty_rationale",""),
            probability_rationale       = nd.get("probability_rationale",""),
            reconciliation_flags        = nd.get("reconciliation_flags",""),
            confidence                  = float(nd.get("confidence",0.7))
        )
        sources = [SourceRef(
            title=s.get("title",""), publisher=s.get("publisher",""),
            url=s.get("url",""), date=s.get("date",""),
            relevance=float(s.get("relevance",0.7)), stance=s.get("stance","neutral")
        ) for s in d.get("sources",[])]
        seeded.append(SeededEvent(
            id=d["id"], label=d.get("label",d["id"]),
            description=d.get("description",""),
            temporal_status=d.get("temporal_status","unresolved"),
            probability=float(d.get("probability",0.5)),
            time_estimate=d.get("time_estimate"),
            time_uncertainty=float(d.get("time_uncertainty",0.5)),
            categories=d.get("categories",{}),
            sources=sources,
            source_count=d.get("source_count",len(sources)),
            source_agreement=float(d.get("source_agreement",0.5)),
            recency=float(d.get("recency",0.5)),
            branch_group=d.get("branch_group"),
            branch_label=d.get("branch_label"),
            causal_candidates=d.get("causal_candidates",[]),
            downstream_impact=float(d.get("downstream_impact",0.5)),
            disruption_score=float(d.get("disruption_score",0.5)),
            novelty=float(d.get("novelty",0.5)),
            causal_support=float(d.get("causal_support",0.5)),
            extraction_notes=diag
        ))
    return seeded


def infer_branch_groups_and_causal_candidates_specialized(
    event_candidates: list,
    source_summaries: str,
    topic: str = ""
) -> dict:
    """
    Stage C: Specialized relational structure inference.

    Receives all candidate events with scores and Stage A hints.
    Returns branch_groups and causal_edges as a structured dict.

    This stage does the structural work — branch grouping and edge typing —
    that requires seeing the whole event field at once.

    In UOT terms: branch groups are properties of the event field,
    not of isolated events. A branch exists when multiple possible
    determinations compete for the same unresolved resolution point.

    In stub mode: returns the pre-built structure for the test scenario.
    """
    if not LIVE_MODE:
        # Stub: return the structure that the Trump test scenario should have.
        # In live mode this comes from the AI.
        return {
            "branch_groups": [
                {
                    "branch_group": "midterms_2026",
                    "resolution_question": "Who controls Congress after the 2026 midterms?",
                    "members": [
                        {"event_id": "e3a", "branch_label": "D_house_only"},
                        {"event_id": "e3b", "branch_label": "D_sweep"},
                        {"event_id": "e3c", "branch_label": "R_hold"}
                    ],
                    "mutual_exclusivity_confidence": 0.95,
                    "rationale": "These are competing outcomes of the same election; only one can be realized."
                },
                {
                    "branch_group": "maga_succession",
                    "resolution_question": "How does the MAGA movement resolve its succession question post-Trump?",
                    "members": [
                        {"event_id": "e6a", "branch_label": "crisis"},
                        {"event_id": "e6b", "branch_label": "stabilization"},
                        {"event_id": "e6c", "branch_label": "fragmentation"}
                    ],
                    "mutual_exclusivity_confidence": 0.80,
                    "rationale": "Three competing resolutions of the same succession indeterminacy."
                }
            ],
            "causal_edges": [
                {"source_id": "e1",  "target_id": "e2",  "relation_type": "causal",
                 "causal_weight": 0.7, "uncertainty": 0.2, "feedback_strength": 0.0,
                 "rationale": "Serial cabinet firings accelerate approval collapse."},
                {"source_id": "e2",  "target_id": "e3a", "relation_type": "causal",
                 "causal_weight": 0.7, "uncertainty": 0.25, "feedback_strength": 0.0,
                 "rationale": "Approval collapse supports Democratic House gains."},
                {"source_id": "e2",  "target_id": "e3b", "relation_type": "causal",
                 "causal_weight": 0.5, "uncertainty": 0.35, "feedback_strength": 0.0,
                 "rationale": "Very low approval also supports Senate flip, but less strongly."},
                {"source_id": "e2",  "target_id": "e3c", "relation_type": "inhibitory",
                 "causal_weight": 0.3, "uncertainty": 0.4, "feedback_strength": 0.0,
                 "rationale": "Approval collapse makes Republican hold structurally unlikely."},
                {"source_id": "e1",  "target_id": "e4",  "relation_type": "causal",
                 "causal_weight": 0.6, "uncertainty": 0.3, "feedback_strength": 0.0,
                 "rationale": "Cabinet chaos signals to allies that US commitments are unreliable."},
                {"source_id": "e3a", "target_id": "e5",  "relation_type": "causal",
                 "causal_weight": 0.7, "uncertainty": 0.35, "feedback_strength": 0.0,
                 "rationale": "D House triggers oversight; executive confrontation follows."},
                {"source_id": "e3b", "target_id": "e5",  "relation_type": "reinforcing",
                 "causal_weight": 0.8, "uncertainty": 0.3, "feedback_strength": 0.0,
                 "rationale": "D Senate + House makes confrontation more certain and broader."},
                {"source_id": "e3c", "target_id": "e5",  "relation_type": "inhibitory",
                 "causal_weight": 0.2, "uncertainty": 0.5, "feedback_strength": 0.0,
                 "rationale": "R hold reduces confrontation probability significantly."},
                {"source_id": "e5",  "target_id": "e6a", "relation_type": "causal",
                 "causal_weight": 0.5, "uncertainty": 0.45, "feedback_strength": 0.0,
                 "rationale": "Constitutional confrontation weakens MAGA's successor prospects."},
                {"source_id": "e3a", "target_id": "e7",  "relation_type": "causal",
                 "causal_weight": 0.5, "uncertainty": 0.4, "feedback_strength": 0.0,
                 "rationale": "D House energizes progressive wing; fracture intensifies."},
                {"source_id": "e7",  "target_id": "e6a", "relation_type": "causal",
                 "causal_weight": 0.4, "uncertainty": 0.5, "feedback_strength": 0.2,
                 "rationale": "Democratic fracture reduces opposition coherence; succession crisis deepens."},
                {"source_id": "e7",  "target_id": "e6b", "relation_type": "inhibitory",
                 "causal_weight": 0.3, "uncertainty": 0.5, "feedback_strength": 0.0,
                 "rationale": "Democratic chaos slightly reduces pressure on MAGA to unify."}
            ]
        }

    # Live mode: call Stage C prompt
    events_with_hints = "\n".join(
        f"- {e['id']}: {e['label']} | P={e.get('probability',0.5):.2f} "
        f"DI={e.get('downstream_impact',0.5):.2f} "
        f"status={e.get('temporal_status','?')} "
        f"hints_branch={e.get('branch_hints',[])} "
        f"hints_causal={e.get('causal_hints',[])}"
        for e in event_candidates
    )
    try:
        result = call_anthropic_api(
            "You are a UOT temporal graph structure analyzer. Return only valid JSON.",
            PROMPT_STAGE_C_STRUCTURE.format(
                topic=topic,
                source_summaries=source_summaries,
                events_with_hints=events_with_hints
            ),
            model="claude-sonnet-4-6",   # Sonnet: branch/causal structure is semantically critical
            model="claude-haiku-4-5-20251001",
            max_tokens=3000,
            timeout=90
        )
        if isinstance(result, dict):
            return result
        # If result is a list, try to interpret it as edges
        if isinstance(result, list):
            return {"branch_groups": [], "causal_edges": result}
    except Exception:
        pass
    # Safe fallback: no inferred structure — seeded graph still usable
    return {"branch_groups": [], "causal_edges": []}


def apply_structure_to_event_candidates(
    event_dicts: list,
    structure
) -> tuple:
    """
    Apply Stage C branch groups and causal edges to event dicts.
    Fully defensive: every AI-returned value is type-checked before .get() is called.
    """
    if not isinstance(structure, dict):
        return list(event_dicts), []

    id_to_event = {e["id"]: e for e in event_dicts if isinstance(e, dict)}
    stage_c_edges = []

    # ── Branch groups ─────────────────────────────────────────────────────────
    for bg in structure.get("branch_groups", []):
        if not isinstance(bg, dict):
            continue
        group_key   = bg.get("branch_group") or bg.get("group_key") or "branch"
        resolution  = bg.get("resolution_question", "")
        confidence  = float(bg.get("mutual_exclusivity_confidence", 0.7) or 0.7)
        rationale   = bg.get("rationale", "")

        members = bg.get("members", [])
        if not isinstance(members, list):
            continue

        for member in members:
            # member may be a dict {"event_id":..., "branch_label":...}
            # or a plain string "e1"
            if isinstance(member, dict):
                eid   = member.get("event_id") or member.get("id")
                label = member.get("branch_label") or member.get("label", group_key)
            elif isinstance(member, str):
                eid   = member
                label = group_key
            else:
                continue

            if eid and eid in id_to_event:
                id_to_event[eid]["branch_group"]  = group_key
                id_to_event[eid]["branch_label"]  = label
                id_to_event[eid]["causal_candidates"] = id_to_event[eid].get(
                    "causal_candidates", [])
                if "extraction_notes" not in id_to_event[eid]:
                    id_to_event[eid]["extraction_notes"] = {}
                id_to_event[eid]["extraction_notes"]["branch_rationale"] = (
                    f"{resolution} | conf={confidence:.2f} | {rationale[:80]}"
                )

    # ── Causal edges ──────────────────────────────────────────────────────────
    for edge in structure.get("causal_edges", []):
        if not isinstance(edge, dict):
            continue
        src_id = edge.get("source_id") or edge.get("source")
        tgt_id = edge.get("target_id") or edge.get("target")
        if not src_id or not tgt_id:
            continue
        if src_id not in id_to_event or tgt_id not in id_to_event:
            continue

        rt  = edge.get("relation_type", "causal")
        cw  = clamp(float(edge.get("causal_weight",    0.5) or 0.5))
        unc = clamp(float(edge.get("uncertainty",      0.3) or 0.3))
        fb  = clamp(float(edge.get("feedback_strength",0.0) or 0.0))

        stage_c_edges.append({
            "source_id": src_id, "target_id": tgt_id,
            "relation_type": rt,
            "causal_weight": cw, "uncertainty": unc,
            "feedback_strength": fb,
            "rationale": edge.get("rationale", ""),
        })

        # Also annotate the source event's causal_candidates
        if "causal_candidates" not in id_to_event[src_id]:
            id_to_event[src_id]["causal_candidates"] = []
        id_to_event[src_id]["causal_candidates"].append(tgt_id)

    return list(id_to_event.values()), stage_c_edges


def collect_and_extract_seeded_events(topic: str, observer_basis: dict) -> List[SeededEvent]:
    """
    Top-level v0.10 seeding pipeline.

    Stage A: Holistic event extraction + branch/causal hints
             (observer encounters unified possibility field; hints embedded in source language)
    Stage B: Specialized local scoring — downstream_impact, disruption_score, novelty
             (observer applies distinct measurement bases to each event)
    Stage C: Relational structure inference — branch groups + causal edges
             (observer resolves relational field; branches are field-level, not event-level)
    Stage D: Reconciliation — auditing, not discovery
             (observer restores global coherence; Stage C owns structure, Stage D checks it)
    Stage E: Return SeededEvent list for user-observer review before simulation

    In UOT terms:
      A = perceive   B = measure   C = resolve structure
      D = restore coherence   E = observer confirmation

    In stub mode (LIVE_MODE=False): returns empty list.
    Use build_seeded_test_world_v10() for the pre-built test scenario.
    """
    if not LIVE_MODE:
        print("[STUB] LIVE_MODE=False. Use build_seeded_test_world_v12() for pre-built scenario.")
        return []

    print(f"[v0.12] Extracting for topic: {topic}")

    # Stage 0: query planning + Stage 0.5: source packets
    queries = generate_search_queries(topic, observer_basis)
    print(f"[v0.12] Stage 0: {len(queries)} queries ({queries[0]['purpose']} + {len(queries)-1} targeted)")
    source_packets   = web_search_sources(topic, observer_basis)
    source_summaries = format_source_packets_for_stage_a(source_packets)
    src_types = sorted({p.source_type for p in source_packets})
    print(f"         Stage 0.5: {len(source_packets)} SourcePackets ({', '.join(src_types)})")

    # Stage A: holistic extraction + branch/causal hints + evidence gap detection
    print("[v0.12] Stage A: Holistic extraction + hints + evidence gap detection...")
    event_dicts, follow_up_queries, missing_evidence = extract_candidate_events_holistic(
        topic, source_summaries, observer_basis
    )
    print(f"         {len(event_dicts)} events | {len(follow_up_queries)} follow-up queries | {len(missing_evidence)} gaps")

    # Stage 0.6: adaptive follow-up search (if Stage A requested it)
    if follow_up_queries:
        print(f"[v0.12] Stage 0.6: Adaptive follow-up ({len(follow_up_queries)} queries from Stage A)...")
        if missing_evidence:
            for gap in missing_evidence[:2]:
                print(f"         gap: {gap}")
        extra_packets    = web_search_sources(topic, observer_basis, follow_up_queries)
        all_packets      = diversify_and_rank_sources(source_packets + extra_packets)
        source_summaries = format_source_packets_for_stage_a(all_packets)
        print(f"         Expanded to {len(all_packets)} SourcePackets")

    # Stage B: specialized local scoring
    print("[v0.12] Stage B: Specialized scoring (downstream, disruption, novelty)...")
    event_dicts = score_event_fields_specialized(event_dicts, source_summaries)

    # Stage C: relational structure inference
    print("[v0.12] Stage C: Relational structure inference (branch groups + causal edges)...")
    structure = infer_branch_groups_and_causal_candidates_specialized(
        event_candidates=event_dicts,
        source_summaries=source_summaries,
        topic=topic
    )
    event_dicts, stage_c_edges = apply_structure_to_event_candidates(event_dicts, structure)
    n_branches = len(structure.get("branch_groups", []))
    n_edges    = len(structure.get("causal_edges", []))
    print(f"         {n_branches} branch groups, {n_edges} causal edges inferred")

    # Stage D: reconciliation (auditing, not structure discovery)
    print("[v0.12] Stage D: Reconciliation (auditing Stage C structure)...")
    event_dicts = reconcile_scores_for_consistency(event_dicts)

    seeded = build_seeded_events_from_dicts(event_dicts)
    print(f"[v0.12] Complete: {len(seeded)} SeededEvents + {n_edges} edges ready for review")
    return seeded, stage_c_edges


# Alias for API compatibility — the API imports this name
build_seeded_test_world_v12 = build_seeded_test_world

# ============================================================
# Run
# ============================================================

def display_final(world, scores):
    ts = world.temporal_state
    bp = scores["branch_details"]["global_branch_potential"]
    print(f"\n{'='*70}")
    print(f"FINAL STATE — Step 10")
    print(f"Field: {scores['field_stability']}  Instability: {scores['global_instability']}  "
          f"Branch potential: {bp}  Lambda: {scores['lambda_temporal']}")
    print(f"Flux:{ts.temporal_flux:.3f}  Coh:{ts.temporal_coherence:.3f}  "
          f"Indet:{ts.indeterminacy_density:.3f}  Entropy:{ts.temporal_entropy:.3f}")
    print()
    for gname, gdata in scores["branch_details"]["groups"].items():
        print(f"  [{gname}]  entropy:{gdata['entropy']}  branch_score:{gdata['branch_score']}")
        for lbl, prob in gdata["normalized_probabilities"].items():
            bar = '█' * int(prob * 20)
            print(f"    {lbl:22s}: {prob:.3f}  {bar}")
    print()
    print("Non-branch events:")
    for node in world.event_graph.nodes.values():
        if not node.branch_group:
            st = f"[{node.temporal_status[:3].upper()}]"
            src_note = f" ({node.source_count}src, agree={node.source_agreement:.2f})" if node.source_count else ""
            print(f"  {node.id} {st} P={node.probability:.3f}  "
                  f"TE={node.temporal_energy:.3f}  Indet={node.indeterminacy:.3f}{src_note}")
            print(f"       {node.label}")
            if node.auto_estimated:
                ae = node.auto_estimated
                print(f"       [auto-est] TE={ae.get('temporal_energy','?')}  "
                      f"Coh={ae.get('temporal_coherence','?')}  "
                      f"Ent={ae.get('temporal_entropy','?')}  "
                      f"Indet={ae.get('indeterminacy','?')}")


if __name__ == "__main__":
    params = ModelParams()
    print("UOT TEMPORAL EXTRAPOLATION ENGINE — v0.12")
    print("Seeding pipeline: SourcePackets → extract(A) → score(B) → structure(C) → reconcile(D) → review(E) → simulate")
    print("v0.11: Stage A receives structured compressed records, not raw text")
    print("       Full text retained in SourcePacket.raw_text_ref for audit/fallback")
    print("v0.12: Hybrid query fan-out (Stage 0); adaptive Stage A follow-up (Stage 0.6); two-pass diversification")
    print()
    world = build_seeded_test_world()

    print("\nRunning 10-step simulation...\n")
    for step in range(1, 11):
        world, scores = simulation_step(world, dt=0.1, params=params)

    display_final(world, scores)

    print("\n--- ESTIMATION VALIDATION ---")
    print("Comparing auto-estimated vs. v0.3 hand-coded values for key nodes:")
    check_nodes = ["e1", "e2", "e5", "e6a"]
    # v0.6 target ranges (midpoint of expected range)
    v03_reference = {
        "e1":  {"temporal_energy": 0.5, "temporal_coherence": 0.40, "temporal_entropy": 0.5,  "indeterminacy": 0.1},
        "e2":  {"temporal_energy": 0.6, "temporal_coherence": 0.45, "temporal_entropy": 0.6,  "indeterminacy": 0.15},
        "e5":  {"temporal_energy": 0.65,"temporal_coherence": 0.40, "temporal_entropy": 0.65, "indeterminacy": 0.6},
        "e6a": {"temporal_energy": 0.65,"temporal_coherence": 0.40, "temporal_entropy": 0.6,  "indeterminacy": 0.65},
    }
    for nid in check_nodes:
        node = world.event_graph.nodes[nid]
        ae   = node.auto_estimated
        ref  = v03_reference.get(nid, {})
        print(f"\n  {nid}: {node.label}")
        rec_coh = node.auto_estimated.get("record_coherence", "?")
        cs_graph = node.auto_estimated.get("causal_support_graph", "?")
        print(f"    {'record_coherence':22s}: auto={rec_coh}  (epistemic layer — not compared to hand-coded)")
        if cs_graph != "?":
            print(f"    {'causal_support_graph':22s}: auto={cs_graph}  (graph-derived — replaces hand-supplied value)")
            # Use pass-2 values for coherence and energy
            node.auto_estimated["temporal_coherence"] = node.auto_estimated.get("temporal_coherence_pass2", node.auto_estimated["temporal_coherence"])
            node.auto_estimated["temporal_energy"]    = node.auto_estimated.get("temporal_energy_pass2",    node.auto_estimated["temporal_energy"])
        for field in ["temporal_energy", "temporal_coherence", "temporal_entropy", "indeterminacy"]:
            auto = ae.get(field, "?")
            hand = ref.get(field, "?")
            delta = round(abs(float(auto) - float(hand)), 3) if hand != "?" else "—"
            print(f"    {field:22s}: auto={auto}  hand-coded={hand}  delta={delta}")
