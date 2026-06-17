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
from datetime import date, datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import math
import re


# ============================================================
# Stage 0: Model Parameters
# ============================================================

@dataclass

# ═══════════════════════════════════════════════════════════════════════════════
# HorizonConfig — temporal boundary for a run
# Added per GPT v2 architectural guidance (2026-06-04)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HorizonConfig:
    """Defines the temporal field boundary of a run."""
    start_date:    date
    target_date:   date
    horizon_days:  int
    horizon_label: str   # "3 months", "1 year", "custom", etc.

    @classmethod
    def from_label(cls, label: str) -> "HorizonConfig":
        """Create a HorizonConfig from a human-readable label."""
        today = date.today()
        label_map = {
            "3mo":  90,  "3 months":  90,
            "6mo": 180,  "6 months": 180,
            "1yr": 365,  "1 year":   365,
            "2yr": 730,  "2 years":  730,
            "5yr":1825,  "5 years": 1825,
        }
        days = label_map.get(label, 180)
        return cls(
            start_date=today,
            target_date=today + timedelta(days=days),
            horizon_days=days,
            horizon_label=label,
        )

    @classmethod
    def default(cls) -> "HorizonConfig":
        return cls.from_label("6mo")

    @property
    def target_date_str(self) -> str:
        return self.target_date.strftime("%B %d, %Y")

    @property
    def today_str(self) -> str:
        return self.start_date.strftime("%B %d, %Y")

    @property
    def horizon_months(self) -> float:
        return self.horizon_days / 30.0


def steps_from_horizon(horizon_days: int) -> int:
    """Map horizon length to simulation step count."""
    if horizon_days <= 90:   return 6
    if horizon_days <= 180:  return 8
    if horizon_days <= 365:  return 10
    if horizon_days <= 730:  return 14
    return 18


def normalize_temporal_status(event: dict,
                               today: date,
                               horizon: "HorizonConfig") -> dict:
    """
    Post-extraction temporal normalization.
    Classifies event temporal_status using extracted date strings
    relative to today and the horizon boundary.

    Called after Stage A and again after Stage D.
    """
    if not isinstance(event, dict):
        return event

    # Try to parse a date from various possible fields
    raw_date = (event.get("time_estimate_date") or
                event.get("expected_date") or
                event.get("time_estimate_label") or "")

    event_date = _parse_event_date(str(raw_date))

    if event_date is None:
        # No parseable date — trust the AI's classification
        return event

    current_status = event.get("temporal_status", "unresolved")

    if event_date < today:
        event["temporal_status"]  = "resolved"
        event["record_confidence"] = event.get("probability", 1.0)
        event["probability"]       = 1.0   # event occurred
    elif today <= event_date <= horizon.target_date:
        if current_status not in ("resolved", "counterfactual"):
            event["temporal_status"] = "unresolved"
    elif event_date > horizon.target_date:
        event["temporal_status"] = "beyond_horizon"
        # Beyond-horizon events: reduce energy, increase indeterminacy
        days_beyond = (event_date - horizon.target_date).days
        penalty = min(days_beyond / 365.0, 1.0)
        if "temporal_energy" in event:
            event["temporal_energy"] = float(event["temporal_energy"]) * (1.0 - 0.6 * penalty)
        if "indeterminacy" in event:
            event["indeterminacy"] = min(1.0, float(event["indeterminacy"]) + 0.2 * penalty)
        if "metadata" not in event or not isinstance(event.get("metadata"), dict):
            event["metadata"] = {}
        event["metadata"]["outside_horizon"] = "true"

    return event


def apply_horizon_weighting(node, horizon: "HorizonConfig") -> None:
    """
    Adjusts node field values based on temporal position relative to the horizon.
    Implements GPT / Temporal Extrapolation Algorithms guidance:
      - Past (resolved): reduce energy, reduce indeterminacy
      - Inside horizon: normal simulation zone, no adjustment
      - Beyond horizon: reduce energy, increase indeterminacy as distance penalty

    Modifies node in-place. Call after build_graph_from_estimates().
    """
    status = getattr(node, 'temporal_status', 'unresolved')

    if status == 'resolved':
        # Past events: low energy (already collapsed), low indeterminacy
        node.temporal_energy  = node.temporal_energy  * 0.25
        node.indeterminacy    = node.indeterminacy    * 0.50

    elif status == 'beyond_horizon':
        # Beyond horizon: penalty grows with distance
        meta = getattr(node, 'metadata', {}) or {}
        days_beyond = float(meta.get('days_beyond_horizon', 365))
        penalty = min(days_beyond / 365.0, 1.0)
        node.temporal_energy  = node.temporal_energy  * (1.0 - 0.60 * penalty)
        node.indeterminacy    = min(1.0, node.indeterminacy + 0.20 * penalty)
        node.metadata['outside_horizon'] = 'true'

    # Clamp all modified values
    node.temporal_energy = clamp(node.temporal_energy)
    node.indeterminacy   = clamp(node.indeterminacy)


# ── Branch label degeneracy detector ─────────────────────────────────────────
DEGENERATE_CONJUNCTIONS = [" or ", " and/or ", " vs ", " versus ", " / ", " either "]

def validate_branch_labels(event_dicts: list) -> list:
    """
    Stage D: scan branch_label and label fields for degenerate "OR" conjunctions
    that conflate two contradictory outcomes into one branch member.

    Per GPT Phase 3 guidance: Stage C should prevent it, Stage D should detect it,
    the Review Room should expose it.

    Flags affected events with a reconciliation_flags entry.
    """
    flagged = 0
    for ev in event_dicts:
        if not isinstance(ev, dict):
            continue
        branch_label = str(ev.get("branch_label") or "").lower()
        label        = str(ev.get("label") or "").lower()
        check_text   = branch_label + " " + label

        has_degenerate = any(conj in check_text for conj in DEGENERATE_CONJUNCTIONS)
        if has_degenerate:
            flagged += 1
            existing_flags = ev.get("reconciliation_flags") or ""
            ev["reconciliation_flags"] = (
                (existing_flags + " | " if existing_flags else "") +
                "Branch member conflates multiple outcomes — split required before simulation."
            )
            # Also set in extraction_notes if present
            if isinstance(ev.get("extraction_notes"), dict):
                ev["extraction_notes"]["reconciliation_flags"] = ev["reconciliation_flags"]

    if flagged:
        print(f"[Stage D] Branch label validator: {flagged} degenerate label(s) flagged.")
    return event_dicts


def _parse_event_date(raw: str) -> Optional[date]:
    """
    Try to parse a date from a string like "May 2026", "2026-05", "Q2 2026".
    Returns None if unparseable.
    """
    if not raw or raw.strip() in ("", "None", "null", "TBD", "Unknown"):
        return None
    raw = raw.strip()
    import re

    # ISO format: 2026-05-15
    m = re.match(r'(\d{4})-(\d{2})-(\d{2})', raw)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    # Month Year: May 2026, May-2026
    months = {"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
              "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
    m = re.match(r'([A-Za-z]+)[,\-\s]+(\d{4})', raw)
    if m:
        mon = months.get(m.group(1).lower()[:3])
        if mon:
            try:
                return date(int(m.group(2)), mon, 15)
            except ValueError:
                pass

    # Year only: 2026
    m = re.match(r'^(\d{4})$', raw.strip())
    if m:
        try:
            return date(int(m.group(1)), 6, 15)   # mid-year estimate
        except ValueError:
            pass

    # Q1-Q4 YYYY
    m = re.match(r'Q([1-4])\s*(\d{4})', raw, re.IGNORECASE)
    if m:
        q, yr = int(m.group(1)), int(m.group(2))
        mon = {1:2, 2:5, 3:8, 4:11}[q]
        try:
            return date(yr, mon, 15)
        except ValueError:
            pass

    return None



# ═══════════════════════════════════════════════════════════════════════════════
# Iterative Discrepancy Loop — Phase 3
# Implements the "compare, detect, update, repeat" architecture from the
# 2023 Temporal Extrapolation Algorithms document.
# Architecture designed by GPT, implemented by Claude, June 2026.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Discrepancy:
    """A detected self-consistency violation in the temporal field."""
    id:                     str
    type:                   str   # branch_normalization | exclusive_contradiction |
                                  # causal_inconsistency | indeterminacy_collapse |
                                  # energy_entropy_runaway | flux_runaway
    severity:               float  # 0–1
    node_ids:               list
    branch_group:           Optional[str]
    description:            str
    recommended_correction: str
    is_hard:                bool   # hard = must resolve before convergence


@dataclass
class Correction:
    """A field adjustment applied to resolve a discrepancy."""
    discrepancy_id: str
    type:           str
    target:         str   # node_id or branch_group key
    before:         dict
    after:          dict
    rationale:      str


@dataclass
class IterationResult:
    """Complete record of one iteration pass."""
    iteration:             int
    scores_before:         dict
    discrepancies:         list
    near_discrepancies:    list   # severity 0.03–0.05: near-miss signals
    corrections:           list
    scores_after:          dict
    convergence_delta:     dict
    converged:             bool
    forward_steps:         int = 1   # steps per iteration (currently always 1)


# ── Discrepancy detection ─────────────────────────────────────────────────────

def _get_branch_groups(graph) -> dict:
    """Derive branch groups from node attributes: {bg_key: [node_id, ...]}."""
    groups = {}
    try:
        nodes = graph.nodes if isinstance(graph.nodes, dict) else {}
        for nid, node in nodes.items():
            bg = getattr(node, 'branch_group', None)
            if bg and isinstance(bg, str):
                groups.setdefault(bg, []).append(nid)
    except Exception:
        pass
    return groups


def _observer_salience(node, observer_basis: dict) -> float:
    """Return salience weight [0.75-1.0] for observer-weighted discrepancy scoring."""
    if not observer_basis:
        return 1.0
    cats = getattr(node, 'categories', {}) or {}
    raw = sum(observer_basis.get(k, 0.0) * v for k, v in cats.items())
    normalised = min(1.0, raw / max(1e-6, sum(observer_basis.values())))
    return 0.75 + 0.25 * normalised


def detect_discrepancies(world, scores: dict, observer_basis: dict) -> list:
    """
    Detect self-consistency violations in the current field state.
    Returns a list of Discrepancy objects ordered by severity (highest first).
    """
    graph = world.event_graph
    nodes = graph.nodes
    edges = graph.edges if isinstance(graph.edges, dict) else {}
    discrepancies = []
    did = [0]

    def next_id(prefix):
        did[0] += 1
        return f"{prefix}_{did[0]}"

    branch_groups = _get_branch_groups(graph)

    # ── 1. Branch normalization (HARD) ────────────────────────────────────────
    # Skip single-member groups — consistent with normalize_branch_groups fix.
    # A single-member group always "sums to less than 1" by design (it is not
    # a true branch set), so detecting it as a violation would force it to 1.0.
    for bg_key, member_ids in branch_groups.items():
        if len(member_ids) <= 1:
            continue   # single-member group — not a normalization violation
        total = sum(getattr(nodes[mid], 'probability', 0.0)
                    for mid in member_ids if mid in nodes)
        deviation = abs(total - 1.0)
        if deviation > 0.05:
            discrepancies.append(Discrepancy(
                id=next_id("bn"), type="branch_normalization",
                severity=min(1.0, deviation * 4),
                node_ids=member_ids, branch_group=bg_key,
                description=f"Branch {bg_key} sums to {total:.3f} (deviation {deviation:.3f})",
                recommended_correction="renormalize",
                is_hard=True,
            ))

    # ── 2. Exclusive branch contradiction (HARD) ─────────────────────────────
    for bg_key, member_ids in branch_groups.items():
        members = [(mid, nodes[mid]) for mid in member_ids if mid in nodes]
        high = [(mid, n) for mid, n in members
                if getattr(n, 'probability', 0) > 0.65
                and getattr(n, 'temporal_coherence', 0) > 0.5
                and getattr(n, 'temporal_status', '') not in ('resolved', 'counterfactual')]
        if len(high) >= 2:
            saliences = [_observer_salience(n, observer_basis) for _, n in high]
            avg_sev = sum(saliences) / len(saliences)
            raw_excess = sum(getattr(n,'probability',0) for _,n in high) - 1.0
            sev = max(0.0, min(1.0, avg_sev * raw_excess))
            if sev < 0.05:
                continue  # not a real contradiction
            discrepancies.append(Discrepancy(
                id=next_id("ec"), type="exclusive_contradiction",
                severity=sev,
                node_ids=[mid for mid, _ in high], branch_group=bg_key,
                description=f"Branch {bg_key}: {len(high)} mutually exclusive members are simultaneously high-probability/coherent",
                recommended_correction="reduce_weaker_coherence",
                is_hard=True,
            ))

    # ── 3. Causal inconsistency (HARD) ────────────────────────────────────────
    for eid, edge in edges.items():
        src_id = getattr(edge, 'source_id', None)
        tgt_id = getattr(edge, 'target_id', None)
        if not src_id or not tgt_id or src_id not in nodes or tgt_id not in nodes:
            continue
        src = nodes[src_id]
        tgt = nodes[tgt_id]
        rel = getattr(edge, 'relation_type', 'causal')
        ew  = clamp(getattr(edge, 'causal_weight', 0.5))
        sp  = getattr(src, 'probability', 0.5)
        tp  = getattr(tgt, 'probability', 0.5)
        tgt_status = getattr(tgt, 'temporal_status', 'unresolved')

        if tgt_status in ('resolved', 'counterfactual', 'beyond_horizon'):
            continue

        if rel == 'inhibitory' and sp > 0.75:
            expected_max = 1.0 - ew * sp * 0.6
            if tp > expected_max + 0.20:
                sal = _observer_salience(tgt, observer_basis)
                discrepancies.append(Discrepancy(
                    id=next_id("ci_inh"), type="causal_inconsistency",
                    severity=min(1.0, (tp - expected_max) * 2 * sal),
                    node_ids=[src_id, tgt_id], branch_group=None,
                    description=f"{src_id} inhibits {tgt_id} (weight {ew:.2f}) but target P={tp:.2f} is too high",
                    recommended_correction="reduce_inhibited_probability",
                    is_hard=True,
                ))

        if rel in ('causal', 'reinforcing', 'enabling') and sp > 0.75:
            expected_min = ew * sp * 0.25
            if tp < expected_min - 0.15:
                sal = _observer_salience(tgt, observer_basis)
                discrepancies.append(Discrepancy(
                    id=next_id("ci_rei"), type="causal_inconsistency",
                    severity=min(1.0, (expected_min - tp) * 2 * sal),
                    node_ids=[src_id, tgt_id], branch_group=None,
                    description=f"{src_id} reinforces {tgt_id} (weight {ew:.2f}) but target P={tp:.2f} is implausibly low",
                    recommended_correction="raise_reinforced_probability",
                    is_hard=True,
                ))

    # ── 4. Indeterminacy collapse without evidence (HARD) ────────────────────
    for nid, node in nodes.items():
        status = getattr(node, 'temporal_status', 'unresolved')
        if status in ('resolved', 'counterfactual', 'beyond_horizon'):
            continue
        ind = getattr(node, 'indeterminacy', 0.0)
        prob = getattr(node, 'probability', 0.5)
        if ind > 0.5 and (prob < 0.07 or prob > 0.93):
            sal = _observer_salience(node, observer_basis)
            sev = min(1.0, ind * abs(prob - 0.5) * 2 * sal)
            discrepancies.append(Discrepancy(
                id=next_id("ic"), type="indeterminacy_collapse",
                severity=sev,
                node_ids=[nid], branch_group=None,
                description=f"{nid} has high indeterminacy ({ind:.2f}) but probability collapsed to {prob:.3f}",
                recommended_correction="restore_indeterminacy",
                is_hard=True,
            ))

    # ── 5. Energy-entropy imbalance (SOFT) ───────────────────────────────────
    prev_flux = scores.get('prev_temporal_flux', None)
    curr_flux = scores.get('temporal_flux', 0.0)
    if prev_flux is not None and curr_flux > prev_flux * 1.4 and curr_flux > 0.6:
        discrepancies.append(Discrepancy(
            id=next_id("ee"), type="energy_entropy_runaway",
            severity=min(1.0, (curr_flux - prev_flux) / max(prev_flux, 0.01)),
            node_ids=[], branch_group=None,
            description=f"Temporal flux rising: {prev_flux:.3f} → {curr_flux:.3f}",
            recommended_correction="damp_high_energy_nodes",
            is_hard=False,
        ))

    discrepancies.sort(key=lambda d: d.severity, reverse=True)
    return discrepancies


def extract_near_discrepancies(discrepancies: list) -> list:
    """
    Return near-discrepancy signals in the widened diagnostic band 0.04–0.10.
    Per GPT Phase 4 guidance: widened from 0.03–0.05 to 0.04–0.10.
    Categories:
      severity >= 0.10  → hard discrepancy (corrective)
      0.04 <= sev < 0.10 → near discrepancy (diagnostic only, no correction)
      0.02 <= sev < 0.04 → weak signal (not logged here, optional debug)
    Near-discrepancies do NOT trigger corrections — calibration sensors only.
    """
    return [d for d in discrepancies if 0.04 <= d.severity < 0.10]


# ── Corrections ───────────────────────────────────────────────────────────────

CORRECTION_STEP = 0.07   # nudge size — never a large jump

def apply_corrections(world, discrepancies: list, observer_basis: dict) -> tuple:
    """
    Apply field-math corrections for each detected discrepancy.
    Corrections are modest nudges, not overwrites.
    Returns (updated_world, list_of_corrections).
    """
    graph = world.event_graph
    nodes = graph.nodes
    edges = graph.edges if isinstance(graph.edges, dict) else {}
    corrections = []

    for disc in discrepancies:
        before = {}
        after  = {}

        # ── Branch normalization ──────────────────────────────────────────────
        if disc.type == "branch_normalization" and disc.branch_group:
            member_ids = disc.node_ids
            if len(member_ids) <= 1:
                continue   # never normalize single-member groups
            members = [(mid, nodes[mid]) for mid in member_ids if mid in nodes]
            total = sum(getattr(n, 'probability', 0.0) for _, n in members)
            if total > 0.001:
                for mid, node in members:
                    old_p = getattr(node, 'probability', 0.5)
                    before[mid] = old_p
                    node.probability = clamp(old_p / total)
                    after[mid] = node.probability
            corrections.append(Correction(
                discrepancy_id=disc.id, type="renormalize",
                target=disc.branch_group, before=before, after=after,
                rationale=f"Renormalized branch group — was summing to {total:.3f}"
            ))

        # ── Exclusive branch contradiction ────────────────────────────────────
        elif disc.type == "exclusive_contradiction" and disc.branch_group:
            member_ids = disc.node_ids
            members = [(mid, nodes[mid]) for mid in member_ids if mid in nodes]
            # Sort by probability; reduce coherence of all but the strongest
            members.sort(key=lambda x: getattr(x[1], 'probability', 0), reverse=True)
            for i, (mid, node) in enumerate(members[1:], 1):
                old_c = getattr(node, 'temporal_coherence', 0.5)
                old_p = getattr(node, 'probability', 0.5)
                before[mid] = {'coherence': old_c, 'probability': old_p}
                node.temporal_coherence = clamp(old_c - CORRECTION_STEP * disc.severity)
                node.probability        = clamp(old_p - CORRECTION_STEP * 0.5 * disc.severity)
                after[mid] = {'coherence': node.temporal_coherence, 'probability': node.probability}
            # Re-normalize
            total = sum(clamp(getattr(nodes[mid], 'probability', 0.0))
                        for mid in disc.node_ids if mid in nodes)
            if total > 0.001:
                for mid in disc.node_ids:
                    if mid in nodes:
                        nodes[mid].probability = clamp(nodes[mid].probability / total)
            corrections.append(Correction(
                discrepancy_id=disc.id, type="reduce_weaker_coherence",
                target=disc.branch_group, before=before, after=after,
                rationale="Reduced coherence and probability of weaker branch members to resolve exclusive contradiction"
            ))

        # ── Causal inconsistency ──────────────────────────────────────────────
        elif disc.type == "causal_inconsistency":
            if len(disc.node_ids) >= 2:
                src_id, tgt_id = disc.node_ids[0], disc.node_ids[1]
                if tgt_id in nodes:
                    tgt = nodes[tgt_id]
                    old_p = getattr(tgt, 'probability', 0.5)
                    before[tgt_id] = old_p
                    step = CORRECTION_STEP * disc.severity
                    if 'inhibit' in disc.recommended_correction:
                        tgt.probability = clamp(old_p - step)
                    else:
                        tgt.probability = clamp(old_p + step)
                    after[tgt_id] = tgt.probability
                    corrections.append(Correction(
                        discrepancy_id=disc.id, type=disc.recommended_correction,
                        target=tgt_id, before=before, after=after,
                        rationale=disc.description
                    ))

        # ── Indeterminacy collapse ────────────────────────────────────────────
        elif disc.type == "indeterminacy_collapse":
            nid = disc.node_ids[0] if disc.node_ids else None
            if nid and nid in nodes:
                node = nodes[nid]
                old_p = getattr(node, 'probability', 0.5)
                old_i = getattr(node, 'indeterminacy', 0.5)
                before[nid] = {'probability': old_p, 'indeterminacy': old_i}
                # Pull probability back toward 0.5, restore some indeterminacy
                pull = CORRECTION_STEP * disc.severity
                node.probability   = clamp(old_p + pull * (0.5 - old_p))
                node.indeterminacy = clamp(old_i + CORRECTION_STEP * 0.5)
                after[nid] = {'probability': node.probability, 'indeterminacy': node.indeterminacy}
                corrections.append(Correction(
                    discrepancy_id=disc.id, type="restore_indeterminacy",
                    target=nid, before=before, after=after,
                    rationale=f"Pulled collapsed probability back toward 0.5; restored indeterminacy"
                ))

        # ── Energy-entropy runaway ────────────────────────────────────────────
        elif disc.type == "energy_entropy_runaway":
            damped = []
            for nid, node in nodes.items():
                te = getattr(node, 'temporal_energy', 0.5)
                cs = getattr(node, 'causal_support', 0.3)
                if te > 0.7 and cs < 0.4:
                    old_te = te
                    node.temporal_energy = clamp(te - CORRECTION_STEP * disc.severity)
                    damped.append(nid)
                    before[nid] = old_te
                    after[nid]  = node.temporal_energy
            if damped:
                corrections.append(Correction(
                    discrepancy_id=disc.id, type="damp_high_energy_nodes",
                    target="field", before=before, after=after,
                    rationale=f"Damped {len(damped)} high-energy low-support nodes to prevent flux runaway"
                ))

    return world, corrections


# ── Convergence checking ──────────────────────────────────────────────────────

def compute_resolution_state(scores: dict, loop_converged: bool,
                              final_discrepancies: list) -> str:
    """
    Classify the field's resolution state per GPT Phase 3 guidance.
    Separates three concepts: field_stability, loop_converged, resolution_state.

    coherent:   loop converged + low instability + no hard discrepancies
    unresolved: did not converge but field is stable — genuine unresolved tension
    strained:   high instability or many hard discrepancies remain
    failed:     loop could not stabilize the field at all
    """
    instability = scores.get('global_instability', 0.5)
    hard_remaining = [d for d in final_discrepancies
                      if isinstance(d, dict) and d.get('is_hard') and d.get('severity', 0) > 0.4]

    # A converged field with zero hard discrepancies is always coherent.
    # Instability level does not override a clean convergence — the loop
    # settled, nothing contradicted, result is valid.
    if loop_converged and len(hard_remaining) == 0:
        return 'coherent'
    elif loop_converged and len(hard_remaining) > 0:
        return 'strained'
    elif not loop_converged and instability <= 0.45:
        return 'unresolved'
    else:
        return 'strained'


def compute_field_signature(scores: dict, graph) -> dict:
    """Compact summary of field state for convergence comparison."""
    nodes = graph.nodes if graph else {}
    prob_variance = 0.0
    if nodes:
        probs = [getattr(n, 'probability', 0.5) for n in nodes.values()]
        mean_p = sum(probs) / len(probs)
        prob_variance = sum((p - mean_p) ** 2 for p in probs) / len(probs)
    return {
        'global_instability':    scores.get('global_instability', 1.0),
        'branch_potential':      scores.get('global_branch_potential', 0.5),
        'temporal_flux':         scores.get('temporal_flux', 0.5),
        'prob_variance':         prob_variance,
    }


def compare_signatures(sig_prev: dict, sig_curr: dict) -> dict:
    """Compute convergence deltas between two field signatures."""
    if not sig_prev:
        return {'branch_potential_delta': 1.0, 'flux_delta': 1.0,
                'instability_delta': 1.0, 'prob_variance_delta': 1.0}
    return {
        'branch_potential_delta': abs(sig_curr['branch_potential'] - sig_prev['branch_potential']),
        'flux_delta':             abs(sig_curr['temporal_flux']     - sig_prev['temporal_flux']),
        'instability_delta':      abs(sig_curr['global_instability']- sig_prev['global_instability']),
        'prob_variance_delta':    abs(sig_curr['prob_variance']     - sig_prev['prob_variance']),
    }


def check_convergence(scores_after: dict, discrepancies: list, delta: dict,
                      target_instability: float = 0.35) -> bool:
    """
    Composite convergence check per GPT Phase 3 guidance.
    Convergence = stable enough + internally consistent + not prematurely collapsed.
    """
    hard_remaining = [d for d in discrepancies
                      if d.is_hard and d.severity > 0.4]
    total_disc_score = sum(d.severity for d in discrepancies)
    return (
        scores_after.get('global_instability', 1.0) <= target_instability
        and delta.get('branch_potential_delta', 1.0) < 0.015
        and delta.get('flux_delta', 1.0) < 0.015
        and total_disc_score < 0.05 * max(1, len(discrepancies) + 1)
        and len(hard_remaining) == 0
    )


# ── Main iterative loop ───────────────────────────────────────────────────────

def run_iterative_simulation(world, params, observer_basis: dict,
                              dt: float = 0.1, max_iterations: int = 5) -> tuple:
    """
    Run the iterative discrepancy loop.

    Each iteration:
      1. Run one forward simulation pass
      2. Detect self-consistency violations
      3. Apply field-math corrections
      4. Re-normalize branch groups
      5. Check composite convergence criteria
      6. Repeat up to max_iterations or until converged

    Returns: (final_world, final_scores, iteration_history)

    This implements the "compare, detect, update, repeat" loop from the
    2023 Temporal Extrapolation Algorithms document.
    The loop aims for coherent uncertainty, not false certainty.
    """
    history = []
    prev_sig = None

    for i in range(max_iterations):
        scores_before = _extract_scores(world)

        # Forward simulation pass
        world, step_scores = simulation_step(world, dt, params)

        # Add prev flux to scores for energy-entropy detection
        if prev_sig:
            step_scores['prev_temporal_flux'] = prev_sig.get('temporal_flux', 0.0)

        # Detect discrepancies
        discs = detect_discrepancies(world, step_scores, observer_basis)

        # Apply corrections
        if discs:
            world, corrs = apply_corrections(world, discs, observer_basis)
            compute_branch_probabilities(world)
            scores_after = _extract_scores(world)
        else:
            corrs = []
            scores_after = step_scores

        # Convergence check
        sig = compute_field_signature(scores_after, world.event_graph)
        delta = compare_signatures(prev_sig, sig)
        converged = check_convergence(scores_after, discs, delta)

        # Early exit: if no discrepancies AND all deltas near-zero for 2+ passes,
        # the field is stably unresolved — further iterations change nothing useful.
        if (not discs and i >= 1
                and delta.get('branch_potential_delta', 1.0) < 0.001
                and delta.get('flux_delta', 1.0) < 0.001
                and delta.get('instability_delta', 1.0) < 0.002):
            converged = True   # stably unresolved — mark converged to stop loop

        # Serialize for storage
        # Detect near-misses (severity 0.03-0.05) for calibration data
        near_discs = extract_near_discrepancies(discs)

        history.append(IterationResult(
            iteration=i + 1,
            scores_before=scores_before,
            discrepancies=[_disc_to_dict(d) for d in discs],
            near_discrepancies=[_disc_to_dict(d) for d in near_discs],
            corrections=[_corr_to_dict(c) for c in corrs],
            scores_after=scores_after,
            convergence_delta=delta,
            converged=converged,
            forward_steps=1,
        ))

        prev_sig = sig
        if converged:
            break

    return world, scores_after, history


def _extract_scores(world) -> dict:
    """Extract key scalar scores from world state, using correct field sources."""
    ts    = getattr(world, 'temporal_state', None)
    graph = getattr(world, 'event_graph', None)
    nodes = getattr(graph, 'nodes', {}) if graph else {}

    # Compute instability from params if available, else from energy average
    try:
        # Call module-level functions directly (same module, no import needed)
        _params = ModelParams()
        scores = compute_instability_score(world, _params)
        gi = scores.get('global_instability', 0.5)
    except Exception:
        energies = [getattr(n, 'temporal_energy', 0.5) for n in nodes.values()]
        gi = sum(energies) / max(len(energies), 1) if energies else 0.5

    # TemporalState carries temporal_flux and branch_potential directly
    flux = getattr(ts, 'temporal_flux', 0.5) if ts else 0.5
    bp   = getattr(ts, 'branch_potential', 0.5) if ts else 0.5

    return {
        'global_instability':      round(gi,   4),
        'temporal_flux':           round(flux,  4),
        'global_branch_potential': round(bp,    4),
    }


def _disc_to_dict(d: Discrepancy) -> dict:
    return {
        'id': d.id, 'type': d.type, 'severity': round(d.severity, 4),
        'node_ids': d.node_ids, 'branch_group': d.branch_group,
        'description': d.description, 'is_hard': d.is_hard,
        'recommended_correction': d.recommended_correction,
    }


def _corr_to_dict(c: Correction) -> dict:
    return {
        'discrepancy_id': c.discrepancy_id, 'type': c.type,
        'target': c.target, 'rationale': c.rationale,
        'before': c.before, 'after': c.after,
    }

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

    branch_group:  Optional[str] = None
    branch_label:  Optional[str] = None
    outcome_role:  str           = "causal_context"  # primary_outcome | status_quo_outcome | branch_outcome | causal_context | evidence_context

    # Phase 6.6/6.7: carry slot-mapping fields through the seeding pipeline.
    # Without these, normalize_and_aggregate_primary_slots's output (including
    # canonical_slot_id, branch_probability, slot_raw_probability, and the
    # inversion/relation metadata in `metadata`) was silently dropped at this
    # SeededEvent boundary and never reached the final EventNode/frontend.
    canonical_slot_id:   Optional[str] = None
    branch_probability:  Optional[float] = None
    slot_raw_probability: Optional[float] = None
    seed_metadata: Dict[str, Any] = field(default_factory=dict)

    # Causal candidate IDs (other SeededEvent IDs this might cause/follow)
    causal_candidates: List[str] = field(default_factory=list)

    # Signals for UOT estimation
    downstream_impact: float  = 0.5  # how many/large are downstream consequences
    disruption_score: float   = 0.5  # social/systemic disruption level
    novelty: float            = 0.5  # how surprising/unprecedented
    causal_support: float     = 0.5  # how strongly supported by causal chain
    extraction_notes: Optional[ExtractionDiagnostics] = None  # v0.9: AI extractor rationale


# ============================================================
# Phase 6.8: SlotAssignment — thin internal dataclass.
# Authoritative record of how one observed event bears on one
# canonical outcome slot. normalize_and_aggregate_primary_slots()
# produces a list of these alongside event_dicts/slot_info.
#
# GPT (Phase 6.8 guidance, point 5): SlotAssignment resolves three
# tensions at once:
#   1. Representative collisions: one event can produce two
#      assignments without cloning the event itself.
#   2. Inversion: assignment-level truth, not event mutation.
#   3. Event-kind weighting: property of evidence contribution,
#      not the node itself.
# The EventNode still carries branch_group/branch_label/
# canonical_slot_id/branch_probability for backwards compatibility,
# but SlotAssignment is the authoritative record.
# ============================================================
@dataclass
class SlotAssignment:
    """
    One way a single observed event bears on one canonical outcome slot.
    A single EventNode may produce multiple SlotAssignments (e.g. when
    one event informs two slots via the slot-view clone path).
    """
    slot_id:            str
    event_id:           str
    source_event_id:    Optional[str]    = None   # if this is a slot-view clone
    relation_to_slot:   str              = "supports"  # supports|opposes|context|meta_signal
    relation_confidence: float           = 0.0
    match_confidence:   float            = 0.0
    engine_score:       float            = 0.0
    accepted_by:        str              = "engine_score"  # "engine_score"|"relation_to_slot"
    event_kind:         str              = "substantive_development"
    raw_probability:    float            = 0.5
    effective_probability: float         = 0.5
    weight:             float            = 1.0
    display_label:      str              = ""   # positive_framing_label or original label
    original_label:     Optional[str]    = None
    original_probability: Optional[float] = None
    is_inverted:        bool             = False
    is_slot_view:       bool             = False  # True for slot-view clones
    use_in_graph_simulation: bool        = True
    rationale:          str              = ""


# ============================================================
# Phase 7.3: SlotInfluence — how non-branch events pressure canonical slots
# without becoming branch members. GPT Phase 7 guidance:
#
#   "Non-branch events should influence canonical outcome slots even when
#    they are not branch members."
#
# Once SlotInfluences exist, raw slot probability is informed by two layers:
#   1. Direct SlotAssignments (the representative and aggregated evidence)
#   2. Contextual SlotInfluences (causal-context events that pressure a slot)
#
# Phase 7.5 (causal coupling) will wire these into the temporal energy field.
# ============================================================
@dataclass
class SlotInfluence:
    """
    How a non-branch observed event bears on a canonical outcome slot.
    Created for events with high slot_affinity or Stage C relation signals
    that did NOT become branch members (causal_context, evidence_context,
    or supporting events for other slots that overlap this one's topic space).
    """
    source_event_id:  str
    slot_id:          str
    relation_to_slot: str    = "context"   # supports | opposes | context
    influence_weight: float  = 0.0
    confidence:       float  = 0.0
    rationale:        str    = ""


# ============================================================
# Phase 7.3 — SlotInfluence Diagnostic Wiring (GPT guidance)
# Compute how non-branch evidence would pressure canonical slots,
# store as diagnostics, but do NOT apply to branch results yet.
# Enable live application with ENABLE_SLOT_INFLUENCE_APPLY=true.
# ============================================================
import math as _math
import os as _os

def _logit_sf(p: float) -> float:
    p = min(0.99, max(0.01, float(p)))
    return _math.log(p / (1 - p))

def _sigmoid_sf(x: float) -> float:
    return 1.0 / (1.0 + _math.exp(-float(x)))

_RELATION_SIGN = {
    "supports":    +1.0,
    "opposes":     -1.0,
    "context":     +0.25,
    "meta_signal": +0.15,
    "irrelevant":   0.0,
}

_INFLUENCE_KIND_WEIGHT = {
    "official_action":      1.0,
    "substantive_development": 0.90,
    "forecast_signal":      0.70,
    "expert_assessment":    0.65,
    "market_signal":        0.55,
    "public_opinion_signal": 0.45,
    "causal_context":       0.30,
    "meta_signal":          0.25,
    "evidence_context":     0.20,
}

def compute_slot_influence_pressures(
    slots,
    raw_slot_probs: dict,
    event_dicts: list,
    slot_influence_records: list,
    k_slot_pressure: float = 0.20,
    apply: bool = False,
) -> list:
    """
    Phase 7.3 (GPT guidance): compute pressure each SlotInfluence would apply
    to canonical slot raw probabilities. Diagnostic-only when apply=False.

    Returns list of per-slot pressure diagnostic dicts.
    """
    ENABLE_APPLY = (apply or
                    _os.environ.get("ENABLE_SLOT_INFLUENCE_APPLY", "").lower() in ("1", "true"))

    # Index influence records by slot_id
    influences_by_slot: dict = {}
    for inf in (slot_influence_records or []):
        sid = getattr(inf, 'slot_id', None) or (inf.get('slot_id') if isinstance(inf, dict) else None)
        if sid:
            influences_by_slot.setdefault(sid, []).append(inf)

    # Index events by id for lookup — handles dict/dataclass and EVT_↔ev_ prefix mismatch
    ev_by_id = {}
    for _ev_item in (event_dicts or []):
        if not _ev_item:
            continue
        if isinstance(_ev_item, dict):
            _ev_id = str(_ev_item.get('id', '') or '')
        else:
            _ev_id = str(getattr(_ev_item, 'id', '') or '')
        if _ev_id:
            ev_by_id[_ev_id] = _ev_item
            ev_by_id[_ev_id.upper()] = _ev_item  # case-insensitive fallback
            # EVT_XXX <-> ev_XXX prefix tolerance (same as Stage C _lookup)
            _eid_l = _ev_id.lower()
            if _eid_l.startswith('evt_'):
                ev_by_id['ev_' + _eid_l[4:]] = _ev_item
            elif _eid_l.startswith('ev_'):
                ev_by_id['evt_' + _eid_l[3:]] = _ev_item

    diagnostics = []
    adjusted_probs = dict(raw_slot_probs)  # copy for potential application

    for slot in (slots or []):
        sid = getattr(slot, 'slot_id', None) or (slot if isinstance(slot, str) else None)
        if not sid:
            continue
        raw_p = float(raw_slot_probs.get(sid, 0.5) or 0.5)
        slot_influences = influences_by_slot.get(sid, [])
        if not slot_influences:
            continue

        logit_delta = 0.0
        applied_list = []

        for inf in slot_influences:
            if isinstance(inf, dict):
                src_ev_id   = inf.get('source_event_id', '')
                rel         = str(inf.get('relation_to_slot', '') or '').lower()
                inf_weight  = float(inf.get('influence_weight', 0.5) or 0.5)
                confidence  = float(inf.get('confidence', 0.5) or 0.5)
                rationale   = str(inf.get('rationale', '') or '')
            else:
                src_ev_id  = getattr(inf, 'source_event_id', '')
                rel        = str(getattr(inf, 'relation_to_slot', '') or '').lower()
                inf_weight = float(getattr(inf, 'influence_weight', 0.5) or 0.5)
                confidence = float(getattr(inf, 'confidence', 0.5) or 0.5)
                rationale  = str(getattr(inf, 'rationale', '') or '')

            _src_id = str(src_ev_id or '')
            src_ev = (ev_by_id.get(_src_id)
                   or ev_by_id.get(_src_id.upper())
                   or ev_by_id.get(_src_id.lower())
                   or {})
            if isinstance(src_ev, dict):
                ev_prob = float(src_ev.get('probability', 0.5) or 0.5)
                ev_kind = str((src_ev.get('metadata') or {}).get('event_kind', '')
                              or src_ev.get('event_kind', '') or '').lower()
            else:
                ev_prob = float(getattr(src_ev, 'probability', 0.5) or 0.5)
                ev_kind = str(getattr(src_ev, 'event_kind', '') or '').lower()

            relation_sign = _RELATION_SIGN.get(rel, 0.0)
            kind_weight   = _INFLUENCE_KIND_WEIGHT.get(ev_kind, 0.50)
            # evidence_signal: how far event prob deviates from neutral (0.5)
            evidence_signal = 2.0 * (ev_prob - 0.5)
            pressure = relation_sign * inf_weight * kind_weight * confidence * evidence_signal
            logit_delta += k_slot_pressure * pressure

            # Structural pressure channel (GPT Phase 7 continuation):
            # Uses status, record_coherence, source_agreement instead of raw probability.
            # Best for active/resolved/contextual conditions where p≈0.5 but field signal is real.
            _ts = str((src_ev.get('metadata') or {}).get('temporal_status', '') or
                      src_ev.get('temporal_status', '') or 'unresolved').lower()
            _status_w = {'resolved': 0.80, 'historical': 0.80, 'active': 0.70,
                         'beyond_horizon': 0.20, 'counterfactual': 0.10}.get(_ts, abs(2*ev_prob - 1))
            _rec_coh = float((src_ev.get('metadata') or {}).get('record_coherence', 0.5)
                             or src_ev.get('record_coherence', 0.5) or 0.5)
            _src_agr = float((src_ev.get('metadata') or {}).get('source_agreement', 0.5)
                             or src_ev.get('source_agreement', 0.5) or 0.5)
            structural_signal = relation_sign * inf_weight * _status_w * _rec_coh * _src_agr * confidence
            structural_delta = k_slot_pressure * structural_signal  # diagnostic only, not applied yet

            applied_list.append({
                "source_event_id": src_ev_id,
                "relation_to_slot": rel,
                "kind_weight": round(kind_weight, 2),
                "evidence_signal": round(evidence_signal, 3),
                "probability_pressure": round(k_slot_pressure * pressure, 4),
                "structural_signal": round(_status_w * _rec_coh * _src_agr, 3),
                "structural_delta": round(structural_delta, 4),
                "rationale": rationale[:100],
            })

        # Clamp total logit delta (probability channel)
        logit_delta = max(-0.35, min(0.35, logit_delta))
        candidate_p = _sigmoid_sf(_logit_sf(raw_p) + logit_delta)
        structural_logit_delta = sum(r.get("structural_delta", 0) for r in applied_list)
        structural_logit_delta = max(-0.35, min(0.35, structural_logit_delta))
        structural_candidate_p = _sigmoid_sf(_logit_sf(raw_p) + structural_logit_delta)

        diag = {
            "slot_id":                         sid,
            "raw_probability_before":          round(raw_p, 4),
            "probability_logit_delta":         round(logit_delta, 4),
            "structural_logit_delta":          round(structural_logit_delta, 4),
            "candidate_probability_after":     round(candidate_p, 4),
            "structural_candidate_probability": round(structural_candidate_p, 4),
            "influence_count":                 len(applied_list),
            "influences_applied":              applied_list,
        }
        diagnostics.append(diag)

        if ENABLE_APPLY and logit_delta != 0.0:
            adjusted_probs[sid] = candidate_p
            print(f"[Phase 7.3] SlotInfluence applied: {sid} "
                  f"{raw_p:.3f} → {candidate_p:.3f} "
                  f"(Δprob={logit_delta:+.3f}, Δstruct={structural_logit_delta:+.3f})")
        else:
            print(f"[Phase 7.3] SlotInfluence diagnostic: {sid} "
                  f"raw={raw_p:.3f} → prob_candidate={candidate_p:.3f} "
                  f"struct_candidate={structural_candidate_p:.3f} "
                  f"(Δprob={logit_delta:+.3f}, Δstruct={structural_logit_delta:+.3f}, "
                  f"{len(applied_list)} influence(s), apply=False)")

    return diagnostics


# ============================================================
# Phase 7 (GPT guidance): SeedingResult — explicit boundary contract.
# Every value that crosses the seeding→simulation→API→frontend boundary
# should be carried in this named object, not scattered across function
# return tuples with implicit positional ordering.
# ============================================================
@dataclass
class SeedingResult:
    """
    Authoritative output of _run_seed_pipeline / collect_and_extract_seeded_events.
    Passed to _upsert_run_artifact and referenced by run_simulation and the API.
    """
    seeded_events:          list      = field(default_factory=list)
    stage_c_edges:          list      = field(default_factory=list)
    slot_assignment_records: list     = field(default_factory=list)
    slot_influence_records:  list     = field(default_factory=list)
    seed_quality:           dict      = field(default_factory=dict)
    warnings:               list      = field(default_factory=list)


# ============================================================
# Phase 5: PrimaryQuestion and OutcomeSlot — canonical outcome space
# ============================================================

@dataclass
class OutcomeSlot:
    """A canonical answer slot for the primary question."""
    slot_id:        str
    label:          str
    description:    str  = ""
    slot_polarity:  str  = "yes"    # yes | no | partial | status_quo | alternative
    slot_kind:      str  = "outcome" # outcome | status_quo | threshold_bucket | timeline_bucket
    synthetic_allowed: bool = False  # Phase 6.5: only True for logically-required complements
    assigned_event_id: str = ""     # set when an organic event is mapped to this slot
    # Phase 6: threshold metadata — optional fields for quantitative boundary conditions
    threshold_metric:   Optional[str]   = None  # e.g. "inflation_rate", "fed_funds_rate"
    threshold_operator: Optional[str]   = None  # e.g. "<=", ">=", ">"
    threshold_value:    Optional[float] = None  # e.g. 2.0
    threshold_upper_value: Optional[float] = None  # for range buckets
    threshold_unit:     Optional[str]   = None  # e.g. "percent", "basis_points"
    threshold_window:   Optional[str]   = None  # e.g. "before end of 2026"
    # Phase 7 (GPT guidance): structured threshold bounds and natural-language aliases
    lower_bound:        Optional[float] = None  # numeric lower bound for threshold bucket
    upper_bound:        Optional[float] = None  # numeric upper bound for threshold bucket
    aliases:            list            = field(default_factory=list)  # natural-language equivalents
    # Phase 7 continuation (GPT guidance): actor/scope/directionality for scope-gate checks
    actor_scope:        str             = ""    # primary actor(s), e.g. "North Korea", "US-EU-China"
    geographic_scope:   str             = ""    # domestic_us | international | regional | global | etc.
    scale_level:        str             = ""    # subnational | national | bilateral | multilateral | global
    directionality:     str             = ""    # continuation | intensification | reversal | breakthrough | decline | uncertain
    action_exemplars:   list            = field(default_factory=list)  # evidence patterns that support this slot
    negative_examples:  list            = field(default_factory=list)  # evidence patterns that do NOT support
    evidence_terms:     list            = field(default_factory=list)  # vocabulary likely to appear in sources


@dataclass
class PrimaryQuestion:
    """Canonical representation of the user's question and its answer space."""
    text:               str
    normalized_question: str = ""
    question_type:      str  = "binary"  # binary | multi_outcome | threshold | timeline | open_scenario
    horizon_label:      str  = ""
    primary_branch_group_id: str = "primary_outcome"
    canonical_slots: List[OutcomeSlot] = field(default_factory=list)


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
        "user_adjusted": {},
        "outcome_role": getattr(candidate, "outcome_role", "causal_context"),
        # Phase 6.6/6.7 carry-through (see SeededEvent field comments)
        "canonical_slot_id":    getattr(candidate, "canonical_slot_id", None),
        "branch_probability":   getattr(candidate, "branch_probability", None),
        "slot_raw_probability": getattr(candidate, "slot_raw_probability", None),
        "seed_metadata":        getattr(candidate, "seed_metadata", {}) or {},
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
    branch_group:     Optional[str] = None
    branch_label:     Optional[str] = None
    canonical_slot_id: Optional[str] = None   # Phase 6: stable slot join key
    branch_probability: Optional[float] = None  # Phase 6.6: normalized branch share (display/tracking only)
    slot_raw_probability: Optional[float] = None  # Phase 6.6: slot's aggregated raw support (GPT's middle layer)
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
    outcome_role: str = "causal_context"  # primary_outcome | status_quo_outcome | branch_outcome | causal_context


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
            outcome_role     = n.get("outcome_role", "causal_context"),
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
            # Phase 6.6/6.7 carry-through (see SeededEvent field comments)
            canonical_slot_id    = n.get("canonical_slot_id"),
            branch_probability   = n.get("branch_probability"),
            slot_raw_probability = n.get("slot_raw_probability"),
            metadata         = {
                "downstream_impact":       str(n.get("downstream_impact", 0.5)),
                "source_causal_evidence":  str(n.get("source_causal_evidence", 0.5)),
                # source_causal_evidence is the original SeededEvent.causal_support,
                # kept separate from graph-derived causal_support_graph so that
                # recompute_coherence_and_energy() never feeds back on itself.
                **(n.get("seed_metadata", {}) or {}),
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
    # Clamp to [0.01, 0.99] before Bayesian update so probabilities of exactly
    # 0 or 1 can be moved by observer evidence. P=1.0 has denom=0; P=0.0 gives
    # raw_new=0 regardless of LR — both are epistemic ceilings that block evidence.
    prior   = max(0.01, min(0.99, node.probability))
    raw_new = prior * effective_lr
    denom   = raw_new + (1.0 - prior)
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


def compute_branch_probabilities(world: WorldTimelineState) -> None:
    """
    Phase 6.6 (GPT guidance): non-mutating replacement for the old
    normalize_branch_groups(). Computes each branch group member's
    normalized share of the group's total RAW probability and writes it
    to node.branch_probability — WITHOUT touching node.probability.

    This preserves the three-layer model:
      - node.probability        = raw independent evidentiary estimate
                                   (never overwritten by branch normalization)
      - node.branch_probability = normalized share within its branch group
                                   (display/tracking only)

    Previously, normalize_branch_groups mutated node.probability directly,
    which (a) corrupted the Events list / node detail display for primary
    branch group members, and (b) fed normalized-share values back into
    compute_temporal_energy_sources's "realization" term on the next
    simulation step, when raw probability should drive that calculation.

    IMPORTANT: Single-member groups are skipped — normalizing them always
    forces a share of 1.0, which collapses indeterminacy spuriously.
    A single-member group is a labeled event, not a true branch set.
    """
    groups: Dict[str, List[EventNode]] = {}
    for node in world.event_graph.nodes.values():
        if node.branch_group is not None:
            groups.setdefault(node.branch_group, []).append(node)
    for group_nodes in groups.values():
        if len(group_nodes) <= 1:
            continue   # single-member group: skip
        total = sum(max(n.probability, 1e-9) for n in group_nodes)
        if total > 0:
            for node in group_nodes:
                node.branch_probability = clamp(node.probability / total)


# Backward-compatible alias retained for any external references.
def normalize_branch_groups(world: WorldTimelineState) -> None:
    """Deprecated (Phase 6.6): use compute_branch_probabilities(). Kept as a
    no-mutation alias so any stray callers don't crash, but does the
    non-mutating computation instead of overwriting node.probability."""
    compute_branch_probabilities(world)


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
        # Phase 6.7: slot-view clones (created when one event represents
        # multiple canonical slots, see normalize_and_aggregate_primary_slots
        # Step 6) are display/snapshot views, not independent causal entities.
        # Their temporal_energy stays frozen at its seeded value.
        if str((node.metadata or {}).get('exclude_from_graph_simulation', '')).lower() == 'true':
            next_energies[node_id] = node.temporal_energy
            continue
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
        # Phase 6.7: slot-view clones don't receive independent observer
        # attention/action — their probability stays as seeded. Their
        # branch_probability (computed below via compute_branch_probabilities)
        # still updates via renormalization as other group members change.
        if str((node.metadata or {}).get('exclude_from_graph_simulation', '')).lower() == 'true':
            continue
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
    compute_branch_probabilities(world)
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
        # Phase 6.7.1 superseded the earlier Phase 6.7 fix here: at the time
        # that fix was written, inversion only adjusted slot_raw_probability,
        # leaving node.probability at the event's raw (un-inverted) value — so
        # normalizing from node.probability would have shown the un-inverted
        # number. Phase 6.7.1 changed this: when inversion applies, Step 6 of
        # normalize_and_aggregate_primary_slots now mutates the representative's
        # node.probability (and label) directly to the inverted/positive-framed
        # value AT SEED TIME — so node.probability and slot_raw_probability
        # start out equal for inverted events too.
        #
        # slot_raw_probability is a SEED-TIME SNAPSHOT and is never updated
        # during simulation, while node.probability evolves via
        # apply_observer_effect each step. Normalizing from slot_raw_probability
        # therefore freezes this display at seed-time values, diverging from
        # node.branch_probability (which compute_branch_probabilities computes
        # from current node.probability, and which capture_slot_snapshots reads
        # for the Tracked Outcomes modal) as soon as simulation runs. Normalize
        # from node.probability — correct for inversion (via the seed-time
        # mutation above) AND reflects simulation's evolution, consistent with
        # compute_branch_probabilities.
        probs  = [max(nd.probability, 1e-9) for nd in gnodes]
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

    # Apply horizon-based field weighting if horizon provided
    if horizon is not None:
        for node in graph.nodes.values():
            apply_horizon_weighting(node, horizon)

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

# ── Prediction-market & gambling site blocklist ───────────────────────────────
# These sites produce event-like probability language (e.g. "80% will resolve No")
# that pollutes Stage A with crowd-sourced betting lines, not independent analysis.
PREDICTION_MARKET_DOMAINS = {
    'polymarket.com', 'kalshi.com', 'predictit.org', 'manifold.markets',
    'augur.net', 'betfair.com', 'smarkets.com', 'gjopen.com', 'elicit.org',
}

# ============================================================
# Phase 7 (GPT + UOT guidance): Temporal Observer Protocol
# Shared instruction block prepended to Stage Q, A, and C prompts.
# Shorter version (TEMPORAL_OBSERVER_PROTOCOL_SHORT) for Stage B/D.
# ============================================================
TEMPORAL_OBSERVER_PROTOCOL = """
You are acting as a Temporal Observer inside the UOT Temporal Extrapolation Engine.

Your task is not to predict with certainty. Your task is to observe a scenario field, preserve uncertainty, identify causal structure, and help construct a coherent possibility space for human review.

Follow these principles:

1. Observer Effect — Your interpretation shapes the field. Be explicit about assumptions, avoid hidden judgments, and distinguish observation from inference.

2. Causality — Preserve cause-and-effect structure. Do not confuse causes, outcomes, evidence, background conditions, or meta-signals.

3. Self-Consistency — The field must not contain contradictions. Mutually exclusive outcomes should be separated into branch slots. Supporting, opposing, and contextual evidence should be labeled accurately.

4. Minimal Intervention — Do not force certainty, invent missing outcomes unnecessarily, or collapse a scenario before the evidence supports it. Prefer organic source-grounded events over synthetic completion.

5. Temporal Stability — Favor structures that remain coherent under review and simulation. If the scenario contains genuine unresolved tension, preserve that tension rather than smoothing it away.

Important probability guidance:

Do not treat cautious source language as automatically implying low probability. Journalistic and institutional sources often hedge even when structural evidence is strong. Estimate probability from the full field: source evidence, causal pressure, base-rate context, temporal momentum, observer measurement basis, and unresolved branch structure.

Use cautious wording to adjust confidence, record coherence, or indeterminacy when appropriate. Do not simply convert hedging language into low likelihood.

When evidence is mixed, preserve the tension. When evidence is indirect, mark it as contextual or meta-signal. When an event argues against an outcome, it may still be highly relevant to that outcome and should be represented as opposing evidence rather than rejected.
""".strip()

TEMPORAL_OBSERVER_PROTOCOL_SHORT = """
You are acting as a Temporal Observer inside the UOT Temporal Extrapolation Engine.

Observe carefully, preserve uncertainty, distinguish evidence from outcome, preserve causality, avoid premature collapse, prefer organic evidence, and make assumptions explicit.
""".strip()

PROMPT_HOLISTIC_EXTRACTION = TEMPORAL_OBSERVER_PROTOCOL + """

---

Act as a Temporal Observer identifying events without prematurely collapsing the field.

Extract candidate events that are grounded in source evidence and relevant to the canonical slots. Each event should describe a real development, active condition, plausible future development, or meaningful evidence signal.

Separate:
- primary outcome events,
- status-quo outcome events,
- branch outcome events,
- causal context,
- evidence context,
- meta-signals such as forecasts, polls, prediction markets, or expert projections.

Do not force an event into an outcome slot during extraction. Instead, provide slot affinities and explain which slots the event may bear on.

Also extract, when available:
  entities: list of key actors/organizations/countries mentioned in this event,
  action_terms: list of key verbs/actions/processes this event describes.

Probability guidance (CRITICAL — use four-factor rubric):

Assign probability using four separate internal considerations:
1. Source claim strength — what the source actually says about likelihood, direction, or certainty.
2. Structural base rate — whether the event fits known trajectory, momentum, or institutional inertia.
3. Causal pressure — whether surrounding events support, inhibit, or are neutral toward this event.
4. Horizon fit — whether timing makes the event plausible or certain within the selected scenario window.

Probability = your integrated estimate from all four factors.
Source hedging should mainly affect record_confidence, source_agreement, or indeterminacy — NOT probability directly.

Do not assign low probability merely because sources use cautious language. Hedging reduces confidence, not likelihood.
Do not assign high probability merely because a source sounds confident. Assess structural plausibility independently.

For example: if IEA projects continued oil demand growth but hedges every claim, the structural trajectory (existing infrastructure, policy inertia, current demand trends) determines the probability — not the hedging language. Mark source_agreement as partial and record_confidence as moderate if sources disagree, but keep probability at the structurally supported level.

Identify 5–10 candidate events for a Unified Observer Theory timeline graph from the topic and sources.

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
  probability_rationale, slot_affinities (dict, see CANONICAL OUTCOME SLOTS above if provided),
  expected_window_label (brief string: when this event is expected to resolve),
  horizon_fit (inside|partial|beyond|unclear — see TEMPORAL CONTEXT above if provided).

Also return HINTS (not final structure — Stage C will finalize these):
  branch_hints: list of strings describing possible mutual exclusivity with other events.
    Example: "Mutually exclusive with Democratic sweep and Republican hold outcomes"
  causal_hints: list of strings describing possible causal relationships.
    Example: "Approval collapse increases probability of this outcome"

Do NOT set branch_group, branch_label, or causal_candidates — leave null/empty.
Do NOT estimate downstream_impact, disruption_score, or novelty yet.
Keep descriptions under 80 words. Keep probability_rationale under 30 words.
Aim for 6-8 events maximum to keep response concise.

CRITICAL — probability assignment by outcome_role:
  Events that will be mapped to canonical primary outcome slots (primary_outcome,
  status_quo_outcome) may be treated as mutually exclusive estimates — their
  probabilities may sum to approximately 1.0.
  Events that serve as causal pressure, conditions, or context (causal_context,
  evidence_context) MUST retain independent probabilities. A causal context event
  at 70% probability should show 0.70 regardless of other events. Do NOT compress
  causal context events into a partition that sums to 1.0.

CRITICAL — event label tense rules:
  resolved events: past tense  (e.g. "North Korea Conducted 6th Nuclear Test")
  active events:   present tense or present progressive  (e.g. "Sanctions Regime Remains Active")
  unresolved events: future tense with "Will"  (e.g. "North Korea Will Conduct 7th Nuclear Test")
  counterfactual events: conditional  (e.g. "North Korea Would Have Agreed to Talks")
  Never write an unresolved event in present or past tense — it causes observer confusion.

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
PROMPT_SCORE_COMBINED = TEMPORAL_OBSERVER_PROTOCOL_SHORT + """

---

Act as a Temporal Observer applying disciplined measurement.

Score only the requested property. Do not let importance, novelty, disruption, downstream impact, and likelihood blur into one another.

When estimating:
- downstream impact, ask how many later systems or branches this event could affect.
- disruption, ask how strongly this event destabilizes existing order or continuity.
- novelty, ask how historically unusual or low-precedent this event is.
- probability, ask how strongly the current field supports this event occurring within the horizon.
- confidence, ask how reliable and direct the evidence record is.

Preserve uncertainty where appropriate. A high-impact event may still be unlikely. A likely event may be low-disruption. A cautious source may still describe a high-probability structural trend.

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

PROMPT_RECONCILE = TEMPORAL_OBSERVER_PROTOCOL_SHORT + """

---

Act as a Temporal Observer conducting a stability review.

Audit the field for contradictions, missing complements, degenerate branches, duplicate
outcomes, causal inconsistencies, horizon mismatches, and premature collapse.

Do not erase genuine uncertainty. If the field contains unresolved tension, preserve it
and flag it. The goal is not artificial neatness; the goal is a self-consistent field
that remains honest about uncertainty.

Prefer correction by clarification, remapping, or flagging before correction by invention.

Review these candidate events and scores for internal consistency.
Flag: downstream/disruption/novelty conflation; probability vs source agreement conflicts;
non-mutually-exclusive branch groups; temporal order violations; events needing merge/split;
premature collapse of genuine uncertainty.
Events: {events_with_scores}
Return JSON: {{"event_id": ["flag..."]}}. Empty dict if no flags.
""".strip()



# ============================================================
# Phase 6: Canonical Slot Aggregation — GPT Phase 6 Code Definition
# Events are temporary observations. Canonical slots are persistent possibilities.
# ============================================================

import re as _re_slots

# Phase 6.7 (GPT guidance, point 2): compact synonym-canonicalization map for
# recurring scenario-language equivalences. Deliberately modest — not a general
# thesaurus, just the handful of recurring patterns we've seen cause lexical
# mismatches between slot labels and event labels describing the same concept
# (e.g. "harmonized standards" vs "regulatory convergence").
_SEMANTIC_CANON_MAP = {
    # Phase 6.7: regulatory/convergence vocabulary
    'harmonization': 'converge', 'harmonized': 'converge', 'harmonize': 'converge',
    'alignment': 'converge', 'aligned': 'converge', 'convergence': 'converge',
    'converge': 'converge', 'converging': 'converge',
    'standards': 'framework', 'rules': 'framework', 'framework': 'framework',
    'regulation': 'framework', 'regulations': 'framework', 'regulatory': 'framework',
    'governance': 'framework',
    # Phase 6.7: occurrence/emergence vocabulary
    'materialize': 'occur', 'materializes': 'occur', 'materializing': 'occur',
    'emerge': 'occur', 'emerges': 'occur', 'emerging': 'occur',
    'occur': 'occur', 'occurs': 'occur', 'occurring': 'occur',
    'happen': 'occur', 'happens': 'occur',
    # Phase 6.7: continuation/status-quo vocabulary
    'status': 'continue', 'quo': 'continue', 'continuation': 'continue',
    'continues': 'continue', 'continuing': 'continue', 'continue': 'continue',
    'remain': 'continue', 'remains': 'continue', 'persist': 'continue',
    'persists': 'continue', 'persisting': 'continue', 'ongoing': 'continue',
    'sustained': 'continue', 'maintaining': 'continue', 'maintained': 'continue',
    'trajectory': 'continue',
    # Phase 6.7: failure/collapse vocabulary
    'collapse': 'fail', 'collapses': 'fail', 'fails': 'fail', 'fail': 'fail',
    'failure': 'fail', 'breakdown': 'fail',
    # Phase 7 (added): divergence/fragmentation vocabulary —
    # "regulatory fragmentation" and "divergence into competing regimes"
    # are the same concept but share zero Jaccard tokens without this bridge.
    'divergence': 'diverge', 'diverging': 'diverge', 'diverged': 'diverge',
    'divergent': 'diverge', 'diverges': 'diverge',
    'fragmentation': 'diverge', 'fragmented': 'diverge', 'fragmenting': 'diverge',
    'fragment': 'diverge', 'fragments': 'diverge',
    # persistence variants (complement to the continue group above)
    'persistence': 'continue', 'persistent': 'continue',
    # Phase 7 (morphological variants — the same root with different suffixes
    # scores 0 in Jaccard even though they're the same concept):
    # escalation/escalated/escalatory → escalate
    'escalation': 'escalate', 'escalated': 'escalate', 'escalatory': 'escalate',
    'escalations': 'escalate', 'escalating': 'escalate',
    # destabilization/destabilizing → destabilize
    'destabilization': 'destabilize', 'destabilizing': 'destabilize',
    'destabilizes': 'destabilize', 'destabilized': 'destabilize',
    # provocation/provocative → provoke
    'provocation': 'provoke', 'provocative': 'provoke', 'provocations': 'provoke',
    'provoking': 'provoke', 'provoked': 'provoke',
    # acceleration/accelerating → accelerate
    'acceleration': 'accelerate', 'accelerating': 'accelerate',
    'accelerated': 'accelerate', 'accelerations': 'accelerate',
    # advancement/advancing → advance
    'advancement': 'advance', 'advancing': 'advance', 'advanced': 'advance',
    'advancements': 'advance',
    # stabilization/stabilizing → stabilize
    'stabilization': 'stabilize', 'stabilizing': 'stabilize',
    'stabilized': 'stabilize', 'stabilizes': 'stabilize',
    # normalization/normalizing → normalize
    'normalization': 'normalize', 'normalizing': 'normalize',
    'normalized': 'normalize',
    # resumption/resuming → resume
    'resumption': 'resume', 'resuming': 'resume', 'resumed': 'resume',
    # expansion/expanding → expand
    'expansion': 'expand', 'expanding': 'expand', 'expanded': 'expand',
    'expansions': 'expand',
    # imposition/imposing → impose
    'imposition': 'impose', 'imposing': 'impose', 'imposed': 'impose',
}


def _normalize_semantic_tokens(tokens: set) -> set:
    """Apply _SEMANTIC_CANON_MAP to a token set, mapping recurring synonyms to a
    shared canonical form before similarity comparison."""
    return {_SEMANTIC_CANON_MAP.get(t, t) for t in tokens}


def _slot_tokens(s: str) -> set:
    """Normalize and tokenize a slot label for similarity comparison."""
    s = s.lower()
    for c in '.,!?;:()[]{}': 
        s = s.replace(c, ' ')
    STOP = {'will','may','could','might','would','the','a','an','is','are','was',
            'be','to','of','and','or','in','on','by','for','with','this','that',
            'its','has','have','before','after','within','during','through'}
    return {t for t in s.split() if t and t not in STOP and len(t) > 2}


def _jaccard_sim(a: str, b: str) -> float:
    ta, tb = _slot_tokens(a), _slot_tokens(b)
    if not ta or not tb: return 0.0
    # Phase 6.7: also compute similarity after canonicalizing recurring synonyms,
    # and take the max — catches cases like "harmonized standards" vs "regulatory
    # convergence" without weakening exact-match cases.
    raw_sim = len(ta & tb) / len(ta | tb)
    na, nb = _normalize_semantic_tokens(ta), _normalize_semantic_tokens(tb)
    norm_sim = len(na & nb) / len(na | nb) if na and nb else 0.0
    return max(raw_sim, norm_sim)


def _has_slot_opposition(a: str, b: str) -> bool:
    """Return True if two labels describe opposing outcomes."""
    POS_SIGS = ('resume','reopen','agree','agreement','deal','resolve','reach',
                'pass','approve','escalate','impose','invade','invasion',
                'attack','operation','peak','reform','invoke','normalization')
    NEG_SIGS = ('does not','will not','not ','no ','fail','blocked','reject',
                'collapse','stall','remain unchanged','frozen','status quo',
                'no invasion','no deal','not peak','uninvok','not invok')
    al, bl = a.lower(), b.lower()
    a_pos = any(s in al for s in POS_SIGS)
    a_neg = any(s in al for s in NEG_SIGS)
    b_pos = any(s in bl for s in POS_SIGS)
    b_neg = any(s in bl for s in NEG_SIGS)
    return bool((a_pos and b_neg) or (a_neg and b_pos))


# Phase 6.6.3 (GPT guidance, point 1): regex patterns matching the two common
# sentence constructions for "this outcome will NOT happen":
#   1. "X will not/won't/does not/fails to <verb>"        (negation in the middle)
#   2. "No <subject> will <verb>"                          (negation at the start)
_NEGATION_REGEXES = [
    re.compile(r'\bwill\s+not\s+\w+'),
    re.compile(r"\bwon'?t\s+\w+"),
    re.compile(r'\bdoes\s+not\s+\w+'),
    re.compile(r"\bdoesn'?t\s+\w+"),
    re.compile(r'\bfails?\s+to\s+\w+'),
    re.compile(r'\bwill\s+fail\s+to\s+\w+'),
    re.compile(r'^no\s+\w+.*\bwill\s+\w+'),     # "No Major Agreement Will Emerge"
    re.compile(r'\bremains?\s+unlikely\b'),
]


_NEGATION_STRIP_PATTERNS = [
    (re.compile(r'\bwill\s+not\s+(\w+)', re.IGNORECASE), r'will \1'),
    (re.compile(r"\bwon'?t\s+(\w+)", re.IGNORECASE), r'will \1'),
    (re.compile(r'\bdoes\s+not\s+(\w+)', re.IGNORECASE), r'\1s'),
    (re.compile(r"\bdoesn'?t\s+(\w+)", re.IGNORECASE), r'\1s'),
    (re.compile(r'\bfails?\s+to\s+(\w+)', re.IGNORECASE), r'will \1'),
    (re.compile(r'\bwill\s+fail\s+to\s+(\w+)', re.IGNORECASE), r'will \1'),
]
_NO_X_WILL_Y = re.compile(r'^no\s+(.+?)\s+will\s+(\w+)', re.IGNORECASE)


def _deterministic_positive_framing(ev_label: str) -> str:
    """
    Phase 6.7 (GPT guidance, point 4): when Stage C omits positive_framing_label
    on an "opposes" mapping, fall back to a simple deterministic transformation
    that strips the negation pattern from the event's own label — producing a
    best-effort positive-framing for display. Not a substitute for Stage C's
    semantic judgment, but better than showing nothing or the slot's abstract
    label standing in for the event.

    Examples:
      "Full Trilateral Harmonization Will Not Occur Within Horizon"
        -> "Full Trilateral Harmonization Will Occur Within Horizon"
      "No Major US-EU-China Agreement Will Emerge"
        -> "Major US-EU-China Agreement Will Emerge"
    """
    label = ev_label
    m = _NO_X_WILL_Y.match(label)
    if m:
        rest = m.group(1).strip()
        rest = (rest[0].upper() + rest[1:]) if rest else rest  # capitalize first letter only
        return f"{rest} Will {m.group(2)}"
    for pattern, repl in _NEGATION_STRIP_PATTERNS:
        if pattern.search(label):
            return pattern.sub(repl, label, count=1)
    return label  # no recognized pattern — return unchanged as last resort


def _event_negates_yes_slot(ev_label: str) -> bool:
    """
    Phase 6.6.3 (GPT guidance, point 1): detect when an event's label is a
    clean negation of a "yes"-polarity slot's proposition — e.g. slot says
    "Convergence toward harmonized standards" and the event says "No Major
    Trilateral AI Governance Agreement Will Emerge" or "Full Harmonization
    Will Not Occur." In that case the event's probability (e.g. 78-82%
    chance the negative framing is true) must be inverted before
    contributing to the slot's raw probability — otherwise an event that
    is STRONG EVIDENCE AGAINST an outcome gets counted as strong evidence
    FOR it.

    Deliberately narrow (regex patterns matching "X will not <verb>" and
    "No <subject> will <verb>" constructions specifically) rather than
    _polarity_matches_label's broader NEG_SIGS list (which includes
    "remain"/"frozen"/"status quo" and would misfire on status_quo/no-
    polarity slots whose representative events legitimately describe
    continuation without using these specific negation constructions).
    """
    el = ev_label.lower()
    return any(p.search(el) for p in _NEGATION_REGEXES)


def _polarity_matches_label(slot_polarity: str, label: str) -> bool:
    """Cheap polarity-label agreement check."""
    neg_signals = ('not ','no ','does not','will not','remain','unchanged',
                   'uninvok','frozen','status quo','fail')
    label_l = label.lower()
    label_is_neg = any(s in label_l for s in neg_signals)
    if slot_polarity in ('no', 'status_quo'):
        return label_is_neg
    if slot_polarity == 'yes':
        return not label_is_neg
    return True  # partial / alternative — accept either


# Phase 6.8 (GPT guidance, point 3): event_kind evidence weights.
# Applied INSIDE aggregation (not at branch normalization) so the
# post-simulation display/tracking pipeline isn't affected. When only
# meta/forecast evidence exists, it is still valid but labeled as such
# via evidence_composition. Weighting is relative, not punitive:
# if all evidence for a slot is forecast-based, the aggregation still
# produces a meaningful probability — just with an honest composition label.
# causal_context / evidence_context contribute 0.0 to slot probability
# (they inform causal edge structure, not direct outcome support).
_EVENT_KIND_WEIGHTS = {
    "substantive_development": 1.00,
    "official_action":         0.95,
    "expert_assessment":       0.75,
    "forecast_signal":         0.65,
    "public_opinion_signal":   0.60,
    "market_signal":           0.55,
    "causal_context":          0.00,
    "evidence_context":        0.00,
    # default for unclassified/missing: treated as substantive
}
_META_SIGNAL_WEIGHT_FALLBACK = 0.45  # only used if no direct evidence exists for slot


def aggregate_slot_probability(events: list, effective_probs: dict = None) -> tuple:
    """
    Weighted probability aggregation for a canonical slot.
    Phase 6 base formula:
        weight = 0.45 * record_coherence + 0.30 * source_agreement + 0.25 * source_count_factor
    Synthetics get weight 0.15.

    Phase 6.6.3: effective_probs is an optional {event_id: effective_probability} map
    for inversion-aware aggregation.

    Phase 6.8 (GPT guidance, point 3): event_kind is now a multiplier in the
    evidence-weighting layer alongside record_coherence/source_agreement/source_count.
    Returns (probability, evidence_composition) — callers that only need the float
    can unpack with [0] or use _aggregate_slot_prob_only() below.
    """
    if not events:
        return 0.5, {"event_count": 0, "dominant_evidence_kind": "none",
                     "substantive_weight_share": 0.0, "forecast_weight_share": 0.0,
                     "market_weight_share": 0.0, "official_weight_share": 0.0,
                     "meta_signal_weight_share": 0.0, "other_weight_share": 0.0}
    effective_probs = effective_probs or {}
    weights, probs, kinds, raw_kind_weights = [], [], [], []
    for e in events:
        md = e.get('metadata') or {}
        is_synthetic = str(md.get('synthetic', '')).lower() == 'true'
        ek = str(md.get('event_kind', '') or e.get('event_kind', '') or '').lower()
        kind_multiplier = _EVENT_KIND_WEIGHTS.get(ek, 1.00)  # unknown = substantive
        if is_synthetic:
            w = 0.15
        elif kind_multiplier == 0.0:
            # causal_context / evidence_context: skip for slot probability
            weights.append(0.0)
            probs.append(0.0)
            kinds.append(ek)
            raw_kind_weights.append(0.0)
            continue
        else:
            sc_factor = min(float(e.get('source_count', 0) or 0) / 5.0, 1.0)
            r_coh     = float(e.get('record_coherence', e.get('source_agreement', 0.5)) or 0.5)
            s_agr     = float(e.get('source_agreement', 0.5) or 0.5)
            base_w    = 0.45 * r_coh + 0.30 * s_agr + 0.25 * sc_factor
            w = base_w * kind_multiplier
        weights.append(max(w, 0.05))
        kinds.append(ek or "substantive_development")
        raw_kind_weights.append(kind_multiplier)
        eid = e.get('id')
        if eid in effective_probs:
            probs.append(float(effective_probs[eid]))
        else:
            probs.append(float(e.get('probability', 0.5) or 0.5))

    total_w = sum(weights)
    if total_w == 0:
        return 0.5, {"event_count": len(events), "dominant_evidence_kind": "none",
                     "substantive_weight_share": 0.0, "forecast_weight_share": 0.0,
                     "market_weight_share": 0.0, "official_weight_share": 0.0,
                     "meta_signal_weight_share": 0.0, "other_weight_share": 0.0}

    prob = sum(p * w for p, w in zip(probs, weights)) / total_w

    # Evidence composition: share of total weight by evidence category
    cat_weights = {"substantive": 0.0, "forecast": 0.0, "market": 0.0,
                   "official": 0.0, "meta_signal": 0.0, "other": 0.0}
    for k, w in zip(kinds, weights):
        if k in ("substantive_development",): cat_weights["substantive"] += w
        elif k in ("forecast_signal",):       cat_weights["forecast"] += w
        elif k in ("market_signal",):         cat_weights["market"] += w
        elif k in ("official_action",):       cat_weights["official"] += w
        elif k in ("expert_assessment", "public_opinion_signal"): cat_weights["meta_signal"] += w
        else:                                  cat_weights["other"] += w
    shares = {f"{k}_weight_share": round(v / total_w, 3) for k, v in cat_weights.items()}
    dominant = max(cat_weights, key=cat_weights.get) if any(cat_weights.values()) else "other"

    composition = {
        "event_count": len(events),
        "dominant_evidence_kind": dominant,
        **shares,
    }
    return prob, composition


def _aggregate_slot_prob_only(events: list, effective_probs: dict = None) -> float:
    """Thin wrapper for callers that only need the float (backwards compat)."""
    return aggregate_slot_probability(events, effective_probs)[0]


def _event_text_field(ev: dict) -> str:
    """Combine all text fields on an event for richer semantic matching (GPT Phase 6.5)."""
    parts = [
        str(ev.get('label', '') or ''),
        str(ev.get('description', '') or ''),
    ]
    md = ev.get('metadata') or {}
    for k in ('branch_hint', 'causal_hint', 'source_snippet', 'rationale'):
        v = md.get(k)
        if v: parts.append(str(v))
    return ' '.join(parts)


def _slot_text_field(slot) -> str:
    """Combine slot label, description, and slot_id for richer matching (GPT Phase 6.5)."""
    return ' '.join([
        slot.label or '',
        getattr(slot, 'description', '') or '',
        (slot.slot_id or '').replace('_', ' '),
    ])


# Phase 7 continuation (GPT guidance): direct-negation check for inversion safety.
# Only events that explicitly negate a slot's outcome should trigger inversion.
# "US and EU align without China" does not directly negate "US-EU-China harmonize."
_NEGATION_SIGNALS = frozenset({
    'will not', "won't", 'wont', 'cannot', "can't", 'will never',
    'ruled out', 'impossible', 'no agreement', 'no convergence',
    'collapses', 'fails to', 'not occur', 'not materialize',
    'not emerge', 'not happen', 'no deal', 'no accord',
    'will fail', 'will collapse', 'does not occur', 'does not happen',
    'will not occur', 'will not emerge', 'will not materialize',
    'is not possible', 'is unlikely to', 'without major',
})

def _is_direct_negation(ev_label: str, rationale: str, direct_neg_flag=None) -> bool:
    """
    True if an event directly and explicitly negates a slot's outcome.
    Inversion should only trigger when this returns True.
    GPT Phase 7 continuation: alternative-pathway events must not invert.
    """
    if direct_neg_flag is True:
        return True  # Stage C explicitly flagged this
    text = (str(ev_label or '') + ' ' + str(rationale or '')).lower()
    return any(sig in text for sig in _NEGATION_SIGNALS)


# Phase 7 continuation (GPT guidance): scope compatibility check.
# Actor-scope gate: a US state-level event should not represent an international slot;
# a "capability improvement" event should not represent a "status quo" slot.
def _compute_scope_compatibility(slot, ev: dict) -> float:
    """
    Returns 1.0 if actors/scope are compatible, 0.4-0.7 if mismatched or unclear.
    Used as a pre-check before Gate B and Gate T.
    """
    actor_scope = getattr(slot, 'actor_scope', '') or ''
    scale_level = getattr(slot, 'scale_level', '') or ''
    directionality = getattr(slot, 'directionality', '') or ''

    if not actor_scope and not scale_level:
        return 1.0  # no restriction defined — assume compatible

    # Extract entities from event label + description
    _ev_label = str(ev.get('label', '') or '')
    _ev_desc  = str((ev.get('metadata') or {}).get('description', '') or ev.get('description', '') or '')
    ev_entities = _extract_named_entities(_ev_label + ' ' + _ev_desc)

    # Check actor match
    if actor_scope:
        scope_entities = _extract_named_entities(actor_scope)
        if scope_entities and not (ev_entities & scope_entities):
            # No shared actors — probably a scope mismatch
            # Check if Stage C rationale explicitly bridges the gap
            rationale = str(ev.get('slot_relevance_rationale', '') or '')
            if scope_entities & _extract_named_entities(rationale):
                return 0.7  # Stage C provided a bridge via rationale
            return 0.4  # likely scope mismatch

    # Check directionality compatibility
    if directionality == 'continuation':
        # Status-quo / continuation slots should NOT accept intensification events
        _ev_action_terms = _extract_action_terms(_ev_label)
        intensification_signals = {'escalate', 'expand', 'accelerate', 'increase', 'advance',
                                   'surge', 'breakthrough', 'dramatic', 'unprecedented'}
        if _ev_action_terms & intensification_signals:
            return 0.6  # possible directionality mismatch — not a hard rejection

    return 1.0


# Phase 7 v2: Named entity and action term extraction for topic relevance scoring.
# These replace label_sim as the dominant component, allowing "North Korea ICBM test"
# to match "destabilization_escalation" via shared entities and action classes.
_ENTITY_STOP = frozenset({
    'the','a','an','is','are','was','were','will','may','could','should','would',
    'been','being','have','has','had','do','does','did','not','no','nor','so',
    'yet','but','and','or','if','as','in','on','at','by','for','of','to','with',
    'from','into','through','over','after','before','its','this','that','these',
    'those','their','they','we','you','he','she','who','which','what','when',
    'where','why','how','be','about','without','within','during','between','among',
})
_KNOWN_ENTITIES = frozenset({
    # Geopolitical actors commonly seen in NK / AI Reg / Taiwan / Fed scenarios
    'north','south','korea','taiwan','china','russia','us','usa','eu','uk','nato',
    'un','iran','japan','india','israel','ukraine','france','germany','canada',
    # Financial/economic
    'fed','ecb','imf','wto','g7','g20','opec','brics','oecd','iea','goldman',
    'jpmorgan','citi','morgan','stanley','blackrock',
    # Technology/AI
    'ai','gpt','openai','google','anthropic','meta','deepmind','llm','ml',
    # International relations
    'icbm','nato','iaea','unhcr','asean','sco','quad',
})

def _extract_named_entities(text: str) -> set:
    """
    Phase 7 v2: extract named entities from text using a simple heuristic:
    capitalized non-stop words + a small set of known scenario-relevant entities.
    Returns a set of lowercased entity strings.
    """
    entities = set()
    for word in re.split(r'[\s\-\—]+', text):
        clean = re.sub(r'[^a-zA-Z0-9]', '', word).lower()
        if len(clean) < 2:
            continue
        if clean in _ENTITY_STOP:
            continue
        # Capitalized word (not stop) or known entity
        if (word and word[0].isupper()) or clean in _KNOWN_ENTITIES:
            entities.add(clean)
    return entities


_ACTION_BROAD_STOP = frozenset({
    'will','may','could','might','would','the','a','an','is','are','was','were',
    'be','to','of','and','or','in','on','by','for','with','this','that','its',
    'has','have','before','after','within','during','through','not','no','nor',
    'yet','from','into','over','been','being','their','further','additional',
    'major','significant','full','partial','key','new','recent','current',
    'first','second','next','last','early','late','mid','high','low','but',
    'also','both','about','against','between','more','most','such','than',
    'still','just','even','back','any','each','much','very','well',
})

def _extract_action_terms(text: str) -> set:
    """
    Phase 7 v2: extract action and object terms from text — content words
    after aggressive stop-word removal, with canonical normalization applied.
    Used for action_object_overlap in compute_topic_relevance_score_v2.
    """
    s = text.lower()
    for c in '.,!?;:()[]{}': s = s.replace(c, ' ')
    return {_SEMANTIC_CANON_MAP.get(t, t)
            for t in s.split()
            if t and len(t) > 2 and t not in _ACTION_BROAD_STOP}


def compute_topic_relevance_score(slot, ev: dict, rationale: str = "") -> float:
    """
    Phase 7.2 (GPT guidance): polarity-agnostic topic relevance score.

    Gate A should measure topic relevance, not polarity. The question is:
    "Are this event and this slot about the same actors, object, process,
    threshold, and horizon?" — NOT "Does this event look like the slot's
    proposition?" Polarity (supports/opposes) belongs to Gate B/relation_to_slot.

    Formula (GPT Phase 7.2 spec):
      topic_score =
          0.35 * label_sim        (negation stripped structurally — it's polarity, not topic)
        + 0.25 * description_sim  (full description text, unchanged)
        + 0.15 * slot_affinity    (Stage A's own relevance signal)
        + 0.15 * rationale_overlap (Stage C's stated rationale engages the slot)
        + 0.10 * hint_sim         (slot_id tokens vs event text)

    Phase 7 v2 formula:
      0.25 * entity_overlap     (named entities shared between event and slot)
      0.20 * action_object_overlap (content words after canonical normalization)
      0.15 * slot_affinity      (Stage A slot_affinities field)
      0.15 * rationale_overlap  (Stage C rationale text vs slot vocabulary)
      0.10 * hint_sim           (slot_id tokens vs event text)
      0.10 * horizon_compat     (inside=1.0, partial=0.7, beyond=0.3)
      0.05 * label_sim          (residual Jaccard — floor for schema-rich scenarios)

    Resolved events: evidence-relevance factor (not simulation-energy factor):
      official_action=1.05×, substantive_development=0.90×, context=0.80×.

    Polarity/opposition terms are NOT part of this score. Negation stripping is
    structural preprocessing applied to label before label_sim only.
    """
    label = ev.get('label', '') or ev.get('branch_label', '') or ''
    if not label:
        return 0.0

    # Structural negation strip for label_sim (topic layer, not relation layer)
    _label_for_topic = label
    for _neg_re, _pos_repl in _NEGATION_STRIP_PATTERNS:
        if _neg_re.search(_label_for_topic):
            _label_for_topic = _neg_re.sub(_pos_repl, _label_for_topic, count=1)
            break
    _no_x_m = _NO_X_WILL_Y.match(_label_for_topic)
    if _no_x_m:
        _rest = _no_x_m.group(1).strip()
        _label_for_topic = f"{(_rest[0].upper()+_rest[1:]) if _rest else _rest} Will {_no_x_m.group(2)}"

    slot_text = _slot_text_field(slot)
    ev_text   = _event_text_field(ev)

    # Phase 7 v2 formula (GPT Phase 7.2 spec — entity/action as primary signals):
    # Entity overlap: shared named entities between event and slot texts (including
    # descriptions). Bridges the gap between abstract slot labels ("destabilization
    # escalation") and domain-specific event vocabulary ("North Korea ICBM test").
    _ev_entities   = (_extract_named_entities(label)
                      | _extract_named_entities(ev_text))
    _slot_entities = (_extract_named_entities(slot.label)
                      | _extract_named_entities(slot_text))
    # Include slot aliases, action_exemplars, and evidence_terms for entity/action matching
    for _alias in (getattr(slot, 'aliases', []) or []):
        _slot_entities |= _extract_named_entities(str(_alias))
    for _et in (getattr(slot, 'evidence_terms', []) or []):
        _slot_entities |= _extract_named_entities(str(_et))
    for _ae in (getattr(slot, 'action_exemplars', []) or []):
        _slot_entities |= _extract_named_entities(str(_ae))
    _entity_sim = (len(_ev_entities & _slot_entities) / len(_ev_entities | _slot_entities)
                   if (_ev_entities | _slot_entities) else 0.0)

    # Action/object overlap: content words after canonical normalization — catches
    # morphological variants (escalation/escalated) and semantic synonyms
    _ev_actions   = (_extract_action_terms(_label_for_topic)
                     | _extract_action_terms(ev_text))
    _slot_actions = (_extract_action_terms(slot.label)
                     | _extract_action_terms(slot_text)
                     | _extract_action_terms(slot.slot_id.replace('_', ' ')))
    for _alias in (getattr(slot, 'aliases', []) or []):
        _slot_actions |= _extract_action_terms(str(_alias))
    for _et in (getattr(slot, 'evidence_terms', []) or []):
        _slot_actions |= _extract_action_terms(str(_et))
    for _ae in (getattr(slot, 'action_exemplars', []) or []):
        _slot_actions |= _extract_action_terms(str(_ae))
    _action_sim = (len(_ev_actions & _slot_actions) / len(_ev_actions | _slot_actions)
                   if (_ev_actions | _slot_actions) else 0.0)

    affinity  = float((ev.get('slot_affinities') or {}).get(slot.slot_id, 0.0) or 0.0)
    rat_sim   = float(_rationale_overlaps_slot(rationale, slot)) * 0.5 if rationale else 0.0
    hint_sim  = _jaccard_sim(slot.slot_id.replace('_', ' '), ev_text)
    _hf = str((ev.get('metadata') or {}).get('horizon_fit', '') or '').lower()
    _horiz_sim = 1.0 if _hf in ('inside', '') else 0.7 if _hf == 'partial' else 0.3 if _hf == 'beyond' else 0.5

    # Resolved-event evidence relevance (GPT Phase 7 guidance, point 4)
    _ev_kind_for_sf = str((ev.get('metadata') or {}).get('event_kind', '')
                          or ev.get('event_kind', '') or '').lower()
    _is_resolved = ev.get('temporal_status') in ('resolved', 'historical')
    if _is_resolved and _ev_kind_for_sf == 'official_action':
        _status_factor = 1.05
    elif _is_resolved and _ev_kind_for_sf in ('causal_context', 'evidence_context'):
        _status_factor = 0.80
    elif _is_resolved:
        _status_factor = 0.90
    else:
        _status_factor = 1.00

    # Keep label_sim as a small residual component (0.05) so Jaccard-based
    # matches remain a floor when slot descriptions are generic (future
    # observations with richer Stage Q descriptions will rely more on entities).
    label_sim = _jaccard_sim(slot.label, _label_for_topic)

    base_score = (0.25 * _entity_sim
                  + 0.20 * _action_sim
                  + 0.15 * affinity
                  + 0.15 * rat_sim
                  + 0.10 * hint_sim
                  + 0.10 * _horiz_sim
                  + 0.05 * label_sim) * _status_factor

    return base_score


# Keep old name as alias for any remaining internal callers
def _slot_event_match_score(slot, ev: dict) -> float:
    return compute_topic_relevance_score(slot, ev)


def _rationale_overlaps_slot(rationale: str, slot) -> bool:
    """
    Phase 6.7 (GPT guidance, point 1): check whether Stage C's stated rationale
    for a relation_to_slot classification actually engages with THIS slot's
    proposition — a guard against Stage C asserting a high-confidence relation
    without grounding it in the slot's specific content. Uses the same
    normalized-token similarity as _slot_event_match_score, against the slot's
    label, description, and slot_id.
    """
    if not rationale:
        return False
    slot_text = _slot_text_field(slot)
    sim = max(
        _jaccard_sim(slot.label, rationale),
        _jaccard_sim(slot_text, rationale),
        _jaccard_sim(slot.slot_id.replace('_', ' '), rationale),
    )
    return sim >= 0.10


def find_organic_events_for_slot(slot, event_dicts: list) -> list:
    """
    Find organic (non-synthetic) events that plausibly represent a canonical slot.

    Thresholds:
      >= 0.55: organic match
      0.40-0.55: weak organic match, allowed but flagged (weak_match=True)
      < 0.40: no organic match (slot remains unmapped, NOT rank-assigned)

    outcome_role is now a soft signal, not a hard exclusion — a causal_context
    event that semantically fits a canonical slot can still be promoted.
    """
    candidates = []
    for ev in event_dicts:
        if not isinstance(ev, dict): continue
        md = ev.get('metadata') or {}
        if str(md.get('synthetic', '')).lower() == 'true': continue
        if str(ev.get('id', '')).startswith('EVT_SYNTH'): continue
        if not (ev.get('label') or ev.get('branch_label')): continue

        score = _slot_event_match_score(slot, ev)
        if score >= 0.40:
            candidates.append((score, ev, score < 0.55))  # third elem = weak_match flag

    candidates.sort(key=lambda x: -x[0])
    result = []
    for score, ev, weak in candidates:
        if weak:
            md = ev.setdefault('metadata', {})
            md['weak_match'] = 'true'
            md['weak_match_score'] = round(score, 3)
        result.append(ev)
    return result


def normalize_and_aggregate_primary_slots(
    event_dicts: list,
    primary_question,
    stage_c_structure: dict,
    locked_slots=None,
    prior_slot_probs: dict = None,
) -> tuple:
    """
    Phase 6 core: map events to canonical slots, aggregate probabilities, create synthetics only
    when no organic candidates exist. Returns (modified_event_dicts, slot_info_dict).

    Implements GPT Phase 6 normalize_and_aggregate_primary_slots().
    """
    import uuid as _uuid_agg
    # Always prefer primary_question.canonical_slots (proper OutcomeSlot objects).
    # locked_slots is raw JSON dicts from DB — only use as last resort with conversion.
    if primary_question and getattr(primary_question, 'canonical_slots', None):
        slots = primary_question.canonical_slots
    elif locked_slots:
        # Convert dicts → OutcomeSlot if needed
        raw = locked_slots if isinstance(locked_slots, list) else []
        slots = []
        for s in raw:
            if hasattr(s, 'slot_id'):
                slots.append(s)
            elif isinstance(s, dict):
                slots.append(OutcomeSlot(
                    slot_id=s.get("slot_id","") or "",
                    label=s.get("label","") or s.get("slot_id","").replace("_"," ").title(),
                    slot_polarity=s.get("slot_polarity","yes"),
                    slot_kind=s.get("slot_kind","outcome"),
                ))
            elif isinstance(s, str):
                slots.append(OutcomeSlot(slot_id=s, label=s.replace("_"," ").title(),
                                         slot_polarity="yes", slot_kind="outcome"))
    else:
        slots = []
    if not slots:
        return event_dicts, {}

    bg_id = getattr(primary_question, 'primary_branch_group_id', 'primary_outcome') or 'primary_outcome'
    assignments: dict = {s.slot_id: [] for s in slots}
    # Phase 6.6.3: per-slot effective-probability overrides for events whose
    # label negates a "yes"-polarity slot's proposition (see _event_negates_yes_slot)
    slot_effective_probs: dict = {s.slot_id: {} for s in slots}

    def _register_effective_prob(sid, ev, member=None):
        """
        Phase 6.7 (GPT guidance, point 3): inversion is now relation-aware,
        not just regex-aware. Two independent signals can trigger inversion:
          A. Stage C explicitly marks relation_to_slot="opposes" for this
             (event, slot) mapping (member is the Stage C member dict).
          B. The deterministic negation-pattern detector (_event_negates_yes_slot)
             fires, as a safety net for when Stage C omits relation_to_slot or
             this assignment came from Step 2's fallback search (member=None).

        When inversion fires, richer metadata is stored per GPT's spec:
          metadata.inverted_for_slot      = sid
          metadata.relation_to_slot       = "opposes"
          metadata.original_probability   = event's own raw probability (unchanged)
          metadata.effective_probability  = 1 - original_probability
          metadata.positive_framing_label = Stage C's reframed label, if provided
        """
        slot = next((s for s in slots if s.slot_id == sid), None)
        if not slot: return
        ev_label = str(ev.get('label', '') or '')

        relation = str((member or {}).get('relation_to_slot', '') or '').lower()
        _direct_neg_flag = (member or {}).get('direct_negation', False)
        should_invert = False

        if relation == 'alternative_pathway':
            # GPT Phase 7 continuation: alternative-pathway events must NEVER invert.
            # They describe a different outcome, not a negation of this slot.
            should_invert = False
            print(f"[v0.12] Slot '{sid}': event {(member or {}).get('best_event_id','?')} "
                  f"marked alternative_pathway — no inversion")
        elif relation == 'opposes':
            # Gate: direct negation required for inversion (GPT Phase 7 continuation).
            # Prevent "bilateral US-EU alignment" from inverting "trilateral harmonization."
            _rationale_text = str((member or {}).get('slot_relevance_rationale', '') or '')
            if _is_direct_negation(ev_label, _rationale_text, _direct_neg_flag):
                should_invert = True
            else:
                # Downgrade to alternative_pathway — event opposes via adjacency, not negation
                relation = 'alternative_pathway'
                print(f"[v0.12] Slot '{sid}': opposes downgraded to alternative_pathway "
                      f"(no direct negation signal) for {(member or {}).get('best_event_id','?')}")
        elif slot.slot_polarity in ('yes', 'alternative', 'partial') and _event_negates_yes_slot(ev_label):
            should_invert = True
            if not relation:
                relation = 'opposes'  # inferred, not Stage C-provided

        if should_invert:
            orig = float(ev.get('probability', 0.5) or 0.5)
            eff = max(0.0, min(1.0, 1.0 - orig))
            slot_effective_probs[sid][ev.get('id')] = eff
            md = ev.setdefault('metadata', {})
            md['inverted_for_slot'] = sid
            md['relation_to_slot'] = relation
            md['original_probability'] = round(orig, 3)
            md['effective_probability'] = round(eff, 3)
            pfl = (member or {}).get('positive_framing_label')
            if not pfl:
                # Phase 6.7 (GPT guidance, point 4): deterministic fallback
                # when Stage C omits positive_framing_label on an opposes mapping.
                pfl = _deterministic_positive_framing(ev_label)
            md['positive_framing_label'] = pfl
            print(f"[v0.12] Slot '{sid}': event {ev.get('id')} relation_to_slot="
                  f"'{relation}' — inverting probability {orig:.2f} -> {eff:.2f}")

    # ── Step 1: Use explicit Stage C mappings ──────────────────────────────────
    pbg = stage_c_structure.get("primary_branch_group", {}) if isinstance(stage_c_structure, dict) else {}
    _raw_members = pbg.get("members") or []
    print(f"[v0.12] Stage C primary_branch_group: {len(_raw_members)} members, "
          f"best_event_ids={[m.get('best_event_id', m.get('event_id')) for m in _raw_members[:5]]}, "
          f"match_confidences={[m.get('match_confidence') for m in _raw_members[:5]]}")
    id_to_event = {e["id"]: e for e in event_dicts if isinstance(e, dict)}
    id_to_ci    = {k.upper(): v for k, v in id_to_event.items()}

    def _lookup(eid):
        if not eid: return None
        # Exact match
        r = id_to_event.get(eid)
        if r: return r
        # Case-insensitive via uppercase key dict
        r = id_to_ci.get(str(eid).upper())
        if r: return r
        # Handle EVT_XXX ↔ ev_XXX prefix mismatch (LLM non-determinism in ID format)
        eid_l = str(eid).lower()
        if eid_l.startswith('evt_'):
            alt = 'ev_' + eid_l[4:]
        elif eid_l.startswith('ev_'):
            alt = 'evt_' + eid_l[3:]
        else:
            alt = None
        if alt:
            r = id_to_event.get(alt) or id_to_ci.get(alt.upper())
            if r: return r
        return None

    members_raw = [m for m in (pbg.get("members") or []) if isinstance(m, dict)]
    slot_best_conf: dict = {}   # slot_id -> Stage C's match_confidence for its best_event_id
    slot_by_id = {s.slot_id: s for s in slots}

    for pos, member in enumerate(members_raw):
        slot_id      = str(member.get("slot_id", ""))
        member_label = str(member.get("label", "") or member.get("branch_label", ""))

        # ── New contract (Phase 6.5, Pass 3): candidate_event_ids / best_event_id /
        #    match_confidence. Falls back to legacy event_id for older Stage C
        #    responses or stub mode.
        candidate_ids = member.get("candidate_event_ids")
        best_eid      = member.get("best_event_id", member.get("event_id"))
        match_conf    = member.get("match_confidence")
        if match_conf is None:
            # Legacy response (no match_confidence) — treat presence of event_id
            # as a strong match for backward compatibility.
            match_conf = 1.0 if best_eid else 0.0

        eid = best_eid
        ev  = _lookup(eid) if eid else None

        # Determine which canonical slot this member belongs to
        matched_sid = None

        # 1a. Exact slot_id match
        if slot_id in assignments:
            matched_sid = slot_id

        # 1b. Fuzzy slot_id → slot_id similarity (Stage C often renames slots)
        if not matched_sid:
            best_sim, best_cand = 0.0, None
            for s in slots:
                sim = _jaccard_sim(s.slot_id, slot_id)
                if sim > best_sim:
                    best_sim, best_cand = sim, s.slot_id
            if best_sim >= 0.35:
                matched_sid = best_cand

        # 1c. Member label vs canonical slot label (semantic bridge)
        if not matched_sid and member_label:
            best_sim, best_cand = 0.0, None
            for s in slots:
                sim = max(_jaccard_sim(s.label, member_label),
                          _jaccard_sim(s.slot_id.replace("_", " "), member_label))
                if sim > best_sim:
                    best_sim, best_cand = sim, s.slot_id
            if best_sim >= 0.18:
                matched_sid = best_cand

        # 1d. Event label vs canonical slot label (organic event text as bridge)
        if not matched_sid and ev:
            ev_label = str(ev.get("label", "") or "")
            best_sim, best_cand = 0.0, None
            for s in slots:
                sim = max(_jaccard_sim(s.label, ev_label),
                          _jaccard_sim(s.slot_id.replace("_", " "), ev_label))
                if sim > best_sim:
                    best_sim, best_cand = sim, s.slot_id
            if best_sim >= 0.18:
                matched_sid = best_cand

        if not matched_sid:
            continue  # Position fallback (1e) retired — see Step 2b retirement notes

        # ── Apply match_confidence threshold (Phase 6.5, Pass 3) ──────────────
        # >= 0.55: strong match — assign as primary representative.
        # 0.40-0.55: weak match — assign but flag (Step 6 counts this for organic_rate).
        # < 0.40: Stage C itself was unconfident — treat as no organic event for
        #         this slot from THIS member, but still consider other
        #         candidate_event_ids as secondary contributors to aggregation.
        if ev and match_conf >= 0.40:
            _slot_obj = slot_by_id[matched_sid]
            _member_rationale = str(member.get('slot_relevance_rationale', '') or '')
            _engine_score = compute_topic_relevance_score(_slot_obj, ev, rationale=_member_rationale)

            # Phase 6.7 (GPT guidance, point 1): two-gate acceptance.
            # Gate A: direct lexical/affinity match — unchanged from 6.6.5.
            # Gate B: relation-aware semantic rescue — Stage C may rescue a
            # lexically weak match ONLY when it provides a clear relation_to_slot
            # ("supports" or "opposes") with high relation_confidence, a rationale
            # that demonstrably engages with THIS slot's content, and the event
            # isn't merely a meta-signal/context event standing in for a real one.
            _accept = False
            _accept_reason = None
            if _engine_score >= 0.40:
                _accept = True
                _accept_reason = "engine_score"
            else:
                _relation   = str(member.get('relation_to_slot', '') or '').lower()
                _rel_conf   = member.get('relation_confidence')
                # Fall back to match_confidence when Stage C omits relation_confidence
                if _rel_conf is None or float(_rel_conf or 0.0) == 0.0:
                    _rel_conf = float(match_conf)
                else:
                    _rel_conf = float(_rel_conf)
                _ev_kind    = str(member.get('event_kind', '') or '').lower()
                _rationale  = str(member.get('slot_relevance_rationale', '') or '')

                # Gate T (Phase 7, GPT guidance): threshold_bucket slots use a separate
                # acceptance path. Also handles mutual exclusion: if threshold_bucket_hint
                # identifies a different bucket, mark as alternative_bucket and reject here.
                _slot_kind = getattr(_slot_obj, 'slot_kind', '') or ''
                _slot_aliases = getattr(_slot_obj, 'aliases', []) or []
                _threshold_hint = str(member.get('threshold_bucket_hint', '') or '')
                if _slot_kind == 'threshold_bucket':
                    # Mutual exclusion: reject alternative-bucket events (don't invert them!)
                    if _threshold_hint and _threshold_hint != _slot_obj.slot_id:
                        _accept = False
                        _accept_reason = "alternative_bucket"
                        _b_why.append(f"threshold_hint={_threshold_hint}!={_slot_obj.slot_id} (alternative_bucket)")
                    else:
                        _scope_compat = _compute_scope_compatibility(_slot_obj, member)
                        _threshold_match = (
                            _scope_compat >= 0.60
                            and _rel_conf >= 0.70
                            and _relation in ('supports', 'opposes')
                            and (_engine_score >= 0.14
                                 or any(a.lower() in _rationale.lower() for a in _slot_aliases if a)
                                 or _threshold_hint == _slot_obj.slot_id)
                        )
                        if _threshold_match:
                            _accept = True
                            _accept_reason = "gate_t_threshold"

                # Gate B (Phase 7 continuation, GPT guidance): semantic-confidence override.
                # Scope compatibility is checked first — a mismatched actor/scope
                # cannot be rescued by high semantic confidence alone.
                _scope_compat_b = _compute_scope_compatibility(_slot_obj, member)
                if not _accept:
                    if (_relation in ('supports', 'opposes')
                            and _rel_conf >= 0.75
                            and _engine_score >= 0.16
                            and _scope_compat_b >= 0.60
                            and _ev_kind not in ('meta_signal', 'causal_context', 'evidence_context')
                            and _rationale_overlaps_slot(_rationale, _slot_obj)):
                        _accept = True
                        _accept_reason = f"relation_to_slot={_relation}"

            if not _accept:
                # Phase 7: log conditions sync with ACTUAL Gate B thresholds
                _b_why = []
                if _relation not in ('supports', 'opposes'): _b_why.append(f"relation='{_relation}'")
                if _rel_conf < 0.75: _b_why.append(f"rel_conf={_rel_conf:.2f}<0.75")
                if _engine_score < 0.16: _b_why.append(f"engine_score={_engine_score:.3f}<0.16")
                if _ev_kind in ('meta_signal', 'causal_context', 'evidence_context'): _b_why.append(f"kind={_ev_kind}")
                if _relation in ('supports','opposes') and _rel_conf>=0.70 and _engine_score>=0.22:
                    if not _rationale_overlaps_slot(_rationale, _slot_obj): _b_why.append("rationale_overlap<0.10")
                print(f"[v0.12] Stage C mapped {eid} -> '{matched_sid}' "
                      f"(confidence={match_conf:.2f}) but engine independent "
                      f"score is {_engine_score:.2f} (<0.40) and relation-aware "
                      f"rescue did not apply ({'; '.join(_b_why) or 'unknown'}) - "
                      f"rejecting mapping, slot falls through to Step 2")
            else:
                if match_conf < 0.55 or _engine_score < 0.55:
                    md = ev.setdefault('metadata', {})
                    md['weak_match'] = 'true'
                    md['weak_match_score'] = round(min(float(match_conf), _engine_score), 3)
                ev.setdefault('metadata', {})['stage_c_direct'] = 'true'
                # Carry Stage C's relation/positive-framing info onto the event for
                # this assignment (used by _register_effective_prob and display).
                if member.get('relation_to_slot'):
                    ev.setdefault('metadata', {})[f'relation_to_slot__{matched_sid}'] = member['relation_to_slot']
                if member.get('positive_framing_label'):
                    ev.setdefault('metadata', {})[f'positive_framing_label__{matched_sid}'] = member['positive_framing_label']
                # Phase 6.8: store event_kind from Stage C when provided;
                # otherwise fall back to source-type heuristic so aggregate_slot_probability
                # always has a classification to weight by and evidence_basis_note fires.
                _ek = member.get('event_kind')
                if not _ek:
                    _ev_src = str(ev.get('source_type', '') or '').lower()
                    _ev_lbl = str(ev.get('label', '') or '').lower()
                    if _ev_src in ('forecast', 'polling'):
                        _ek = 'forecast_signal'
                    elif _ev_src == 'market':
                        _ek = 'market_signal'
                    elif any(w in _ev_lbl for w in ('forecast', 'project', 'predict',
                                                      'expect', 'anticipat', 'scenario')):
                        _ek = 'forecast_signal'
                    else:
                        _ek = 'substantive_development'
                ev.setdefault('metadata', {})['event_kind'] = _ek
                assignments[matched_sid].append(ev)
                slot_best_conf[matched_sid] = float(match_conf)
                _register_effective_prob(matched_sid, ev, member)
                print(f"[v0.12] Stage C mapped {eid} -> slot '{matched_sid}' "
                      f"(confidence={match_conf:.2f}, pos={pos}, accept={_accept_reason})")
        elif member.get("no_organic_match_reason"):
            print(f"[v0.12] Stage C: no organic match for slot '{matched_sid}' — "
                  f"{member['no_organic_match_reason']}")

        # Additional candidates: attempt Gate A, then secondary Gate B (stricter: rel_conf >= 0.80)
        # Each slot accepts up to 3 candidates with rank_weight decay.
        _RANK_WEIGHTS = {1: 1.00, 2: 0.55, 3: 0.35}
        if candidate_ids and isinstance(candidate_ids, list):
            _secondary_pos = 2
            for cand_id in candidate_ids[1:]:
                if _secondary_pos > 3:
                    break
                cand_ev = _lookup(cand_id)
                if not cand_ev or cand_ev.get('id') in [e.get('id') for e in assignments[matched_sid]]:
                    continue
                _slot_obj_s = slot_by_id.get(matched_sid)
                _cand_score = _slot_event_match_score(_slot_obj_s, cand_ev) if _slot_obj_s else 0.0
                _cand_accepted = False
                _cand_reason   = ""
                if _cand_score >= 0.40:
                    _cand_accepted = True
                    _cand_reason = f"gate_a({_cand_score:.2f})"
                else:
                    _s_rel   = str(cand_ev.get('relation_to_slot', '') or '').lower()
                    _s_rconf = float(cand_ev.get('relation_confidence') or
                                     cand_ev.get('match_confidence') or 0.0)
                    _s_kind  = str(cand_ev.get('event_kind', '') or '').lower()
                    _s_rat   = str(cand_ev.get('slot_relevance_rationale', '') or '')
                    _s_scope = _compute_scope_compatibility(_slot_obj_s, cand_ev) if _slot_obj_s else 1.0
                    if (_s_rel in ('supports', 'opposes')
                            and _s_rconf >= 0.80
                            and _cand_score >= 0.16
                            and _s_scope >= 0.60
                            and _s_kind not in ('meta_signal', 'causal_context', 'evidence_context')
                            and _slot_obj_s and _rationale_overlaps_slot(_s_rat, _slot_obj_s)):
                        _cand_accepted = True
                        _cand_reason = f"secondary_gate_b(rel={_s_rel},conf={_s_rconf:.2f})"
                    else:
                        print(f"[v0.12] Stage C secondary candidate {cand_id} -> slot "
                              f"'{matched_sid}' rejected (engine score {_cand_score:.2f} < 0.40, "
                              f"secondary Gate B not met)")
                if _cand_accepted:
                    _rank_w = _RANK_WEIGHTS.get(_secondary_pos, 0.35)
                    cand_ev_copy = dict(cand_ev)
                    cand_ev_copy['_rank_weight'] = _rank_w
                    assignments[matched_sid].append(cand_ev_copy)
                    _register_effective_prob(matched_sid, cand_ev)
                    print(f"[v0.12] Stage C secondary candidate {cand_id} -> slot '{matched_sid}' "
                          f"(pos={_secondary_pos}, rank_weight={_rank_w:.2f}, accept={_cand_reason})")
                    _secondary_pos += 1

    # ── Step 2: Fill missing slots via organic matching ───────────────────────
    for slot in slots:
        if not assignments[slot.slot_id]:
            organics = find_organic_events_for_slot(slot, event_dicts)
            assignments[slot.slot_id].extend(organics)
            for _ev in organics:
                _register_effective_prob(slot.slot_id, _ev)

    # ── Step 2b: RETIRED (Phase 6.5, GPT guidance) ────────────────────────────
    # Rank-based fallback assignment was removed. High probability is not the
    # same as semantic fit — assigning the highest-probability remaining event
    # to an unmatched slot produced semantically false mappings (e.g. "North
    # Korea reaches 100-150 warheads" assigned to "Program Reversal" because it
    # was the 5th-ranked event, not because it represents reversal).
    # Slots with no organic match (score >= 0.40) above now remain genuinely
    # unmapped and are handled by Step 3's revised synthetic policy.

    # ── Step 3: Synthetic policy for still-empty slots (GPT Phase 6.5) ────────
    # Synthetics are now rare by construction:
    #   - synthetic_allowed=True slots (binary "no" complement, status_quo,
    #     timeline complement) get a synthetic placeholder with a SMALL
    #     residual prior (0.05-0.10), NOT 0.5 — they no longer compete
    #     on equal footing with evidence-backed organic slots.
    #   - For binary questions, a "no"-complement synthetic instead gets
    #     probability = 1 - (the "yes" slot's organic raw probability),
    #     i.e. it's defined as the logical complement, not an independent guess.
    #   - synthetic_allowed=False slots that found no organic match are left
    #     UNMAPPED — no event is created. They're recorded in slot_info for
    #     audit/diagnostics and the Review Room shows "insufficient evidence".
    unmapped_slots = []
    is_binary = getattr(primary_question, 'question_type', '') == 'binary'

    # Binary complement is symmetric: whichever of the two slots has organic
    # evidence, the OTHER slot — if empty — becomes its logical complement
    # (1 - other_raw), regardless of which side is labeled "yes" or "no".
    binary_mapped_raw  = None   # raw probability of whichever binary slot HAS organics
    binary_empty_slot  = None   # the OutcomeSlot with no organics (the complement)
    if is_binary and len(slots) == 2:
        mapped  = [s for s in slots if assignments[s.slot_id]]
        unmapped = [s for s in slots if not assignments[s.slot_id]]
        if len(mapped) == 1 and len(unmapped) == 1:
            binary_mapped_raw, _ = aggregate_slot_probability(
                assignments[mapped[0].slot_id], effective_probs=slot_effective_probs.get(mapped[0].slot_id))
            binary_empty_slot = unmapped[0]

    for slot in slots:
        if assignments[slot.slot_id]:
            continue

        # Symmetric binary complement overrides synthetic_allowed=False —
        # a 2-slot binary question always needs both sides represented.
        if is_binary and binary_empty_slot is not None and slot.slot_id == binary_empty_slot.slot_id:
            sid = "EVT_SYNTH_" + _uuid_agg.uuid4().hex[:6].upper()
            or_role = "status_quo_outcome" if slot.slot_kind == "status_quo" else "primary_outcome"
            synth_prob = max(0.0, min(1.0, 1.0 - binary_mapped_raw))
            reason = "binary_logical_complement"
            synth = {
                "id": sid, "label": slot.label, "probability": synth_prob,
                "time_estimate": 2, "time_uncertainty": 0.4,
                "temporal_status": "unresolved",
                "branch_group": bg_id, "branch_label": slot.label,
                "outcome_role": or_role, "disruption_score": 0.2,
                "record_coherence": 0.5, "categories": {},
                "canonical_slot_id": slot.slot_id,
                "metadata": {
                    "synthetic": "true", "synthetic_reason": reason,
                    "slot_id": slot.slot_id, "slot_polarity": slot.slot_polarity,
                },
            }
            event_dicts.append(synth)
            id_to_event[sid] = synth
            assignments[slot.slot_id].append(synth)
            print(f"[v0.12] Slot '{slot.slot_id}' — binary complement, "
                  f"created synthetic {sid} (p={synth_prob:.2f})")
            continue

        if not slot.synthetic_allowed:
            unmapped_slots.append(slot.slot_id)
            print(f"[v0.12] Slot '{slot.slot_id}' — no organic match (synthetic_allowed=False), "
                  f"leaving unmapped")
            continue

        sid = "EVT_SYNTH_" + _uuid_agg.uuid4().hex[:6].upper()
        or_role = "status_quo_outcome" if slot.slot_kind == "status_quo" else "primary_outcome"
        # Phase 7 continuation: use series history prior if available.
        # On re-observations, the synthetic placeholder inherits the slot's last
        # observed organic probability rather than defaulting to 8% every time.
        _series_prior = (prior_slot_probs or {}).get(slot.slot_id)
        synth_prob = float(_series_prior) if _series_prior is not None else 0.08
        reason = "logical_complement_residual_prior" if _series_prior is None else "series_prior_inherited"

        synth = {
            "id": sid, "label": slot.label, "probability": synth_prob,
            "time_estimate": 2, "time_uncertainty": 0.4,
            "temporal_status": "unresolved",
            "branch_group": bg_id, "branch_label": slot.label,
            "outcome_role": or_role, "disruption_score": 0.2,
            "record_coherence": 0.5, "categories": {},
            "canonical_slot_id": slot.slot_id,   # stable join key for snapshots
            "metadata": {
                "synthetic": "true",
                "synthetic_reason": reason,
                "slot_id": slot.slot_id, "slot_polarity": slot.slot_polarity,
            },
        }
        event_dicts.append(synth)
        id_to_event[sid] = synth
        assignments[slot.slot_id].append(synth)
        print(f"[v0.12] Slot '{slot.slot_id}' — logical complement, "
              f"created synthetic {sid} (p={synth_prob:.2f}, reason={reason})")

    # ── Step 4: Aggregate raw probabilities (unmapped slots excluded) ─────────
    raw_probs = {}
    slot_evidence_compositions = {}  # Phase 6.8: evidence_composition per slot
    for slot in slots:
        if assignments[slot.slot_id]:
            _prob, _comp = aggregate_slot_probability(
                assignments[slot.slot_id], effective_probs=slot_effective_probs.get(slot.slot_id))
            raw_probs[slot.slot_id] = _prob
            slot_evidence_compositions[slot.slot_id] = _comp

    # ── Step 5: Normalize across MAPPED slots only ────────────────────────────
    total = sum(max(p, 1e-6) for p in raw_probs.values())
    normalized = {sid: (raw_probs[sid] / total if total > 0 else 0.0) for sid in raw_probs}

    # ── Step 6: Write branch fields onto representative events ───────────────
    slot_assignments: list = []  # Phase 6.8: authoritative SlotAssignment records
    # Phase 6.6.1: prevent the SAME event from being chosen as the
    # representative for TWO different slots. This can happen when Stage C
    # returns the same best_event_id for multiple slots (an event genuinely
    # relevant to more than one outcome). Without this guard, the second
    # slot's field mutations (branch_label, canonical_slot_id,
    # branch_probability) overwrite the first's on the shared dict, causing
    # the first slot to silently disappear from the branch group display.
    #
    # Resolution: process slots in DESCENDING order of Stage C's
    # match_confidence for their best_event_id (slot_best_conf). The
    # higher-confidence slot claims its preferred representative first;
    # a lower-confidence slot whose preferred representative was already
    # claimed falls back to its next-best candidate (by record_coherence).
    # Only if a slot has NO unclaimed candidate at all does it fall back to
    # reusing an already-claimed event (rare; logged when it happens).
    # Phase 6.8 (GPT guidance, point 7): seed_quality computed AFTER all
    # gates/inversions/clones/representative selection are final — not
    # accumulated during Step 1 (which lies whenever later gates reject,
    # remap, or synthesize). These are zeroed here and filled in Step 6's
    # post-resolution loop below.
    organic_count   = 0
    synthetic_count = 0
    mapping_methods = {"stage_c_direct": 0, "semantic_match": 0, "weak_semantic_match": 0, "synthetic": 0}
    inverted_count       = 0
    weak_count           = 0
    reused_source_events = 0
    meta_signal_count    = 0
    forecast_dominant_count = 0
    slot_view_count      = 0
    horizon_mismatch_count = 0
    assignment_collision_count = 0

    def _coherence_key(e):
        return (0 if str((e.get('metadata') or {}).get('synthetic','')).lower()=='true'
                else float(e.get('record_coherence', 0.5) or 0.5))

    claim_order = sorted(
        (s for s in slots if assignments[s.slot_id]),
        key=lambda s: slot_best_conf.get(s.slot_id, 0.0),
        reverse=True,
    )
    claimed_rep_ids: set = set()
    slot_rep: dict = {}   # slot_id -> representative event dict

    for slot in claim_order:
        sid = slot.slot_id
        evs = assignments[sid]
        unclaimed = [e for e in evs if e.get('id') not in claimed_rep_ids]
        if unclaimed:
            rep = max(unclaimed, key=_coherence_key)
            claimed_rep_ids.add(rep.get('id'))
        else:
            # Phase 6.7 (GPT guidance, point 5): do NOT reuse the same dict
            # object as another slot's representative — that causes the
            # second-pass mutation collision (one slot's branch_label/
            # canonical_slot_id/branch_probability silently overwrites the
            # other's). A single observed development can legitimately inform
            # multiple canonical slots (e.g. EVT_006 informs both
            # "incremental_advancement" and "status_quo_continuation").
            #
            # Short-term patch (per GPT, pending a full SlotAssignment layer):
            # create a shallow clone — same probability/label/evidence, its
            # own branch_label/canonical_slot_id/branch_probability — marked
            # as a display/snapshot view of the source event, excluded from
            # causal simulation so the same real-world observation doesn't
            # become two causal entities in the graph.
            base = max(evs, key=_coherence_key)
            rep = dict(base)
            rep['metadata'] = dict(base.get('metadata') or {})
            rep['id'] = f"{base.get('id')}__{sid}"
            rep['metadata']['is_slot_view'] = 'true'
            rep['metadata']['source_event_id'] = base.get('id')
            rep['metadata']['exclude_from_graph_simulation'] = 'true'
            event_dicts.append(rep)
            claimed_rep_ids.add(rep['id'])
            print(f"[v0.12] Slot '{sid}': no unclaimed representative available, "
                  f"creating slot-view {rep['id']} from {base.get('id')} "
                  f"(shared evidence, distinct slot)")
        slot_rep[sid] = rep

    for slot in slots:
        sid   = slot.slot_id
        evs   = assignments[sid]
        if not evs: continue
        rep = slot_rep[sid]
        rep["branch_group"]       = bg_id
        rep["branch_label"]       = slot.label
        rep["outcome_role"]       = "status_quo_outcome" if slot.slot_kind == "status_quo" else "primary_outcome"
        rep["canonical_slot_id"]  = sid
        # Phase 6.6 (GPT guidance): do NOT overwrite rep["probability"] — it remains
        # the representative event's own raw/independent measured probability.
        # The branch group's normalized share is stored separately as
        # branch_probability, mirroring how secondary branch groups already work
        # (their members keep independent probabilities; branch share is computed
        # at display time by compute_branch_potential_details).
        rep["branch_probability"] = normalized.get(sid, 0.0)
        rep["slot_raw_probability"] = raw_probs.get(sid, rep.get("probability", 0.5))
        if len(evs) > 1:
            rep["mapped_event_ids"] = [e["id"] for e in evs]
        # Phase 6.8: store evidence_composition on the rep and in metadata
        _ec = slot_evidence_compositions.get(sid)
        if _ec:
            rep["evidence_composition"] = _ec
            _dom = _ec.get("dominant_evidence_kind", "")
            if _dom not in ("substantive", "official", ""):
                rep.setdefault("metadata", {})["evidence_basis_note"] = (
                    f"Evidence basis: mostly {_dom.replace('_', ' ')}."
                )

        # Phase 6.8: build authoritative SlotAssignment record for this slot
        _rep_for_sa = rep
        _sa_md = (_rep_for_sa.get('metadata') or {})
        _sa = SlotAssignment(
            slot_id=sid,
            event_id=_rep_for_sa.get('id', ''),
            source_event_id=_sa_md.get('source_event_id'),
            relation_to_slot=_sa_md.get(f'relation_to_slot__{sid}')
                             or _sa_md.get('relation_to_slot', 'supports'),
            relation_confidence=float(slot_best_conf.get(sid, 0.0) or 0.0),
            match_confidence=float(slot_best_conf.get(sid, 0.0) or 0.0),
            engine_score=float(_sa_md.get('weak_match_score', 0.0) or 0.0),
            accepted_by=_sa_md.get('stage_c_direct') and 'engine_score' or 'semantic_match',
            event_kind=_sa_md.get('event_kind', 'substantive_development'),
            raw_probability=float((_sa_md.get('original_probability')
                                   or _rep_for_sa.get('probability', 0.5)) or 0.5),
            effective_probability=float(_rep_for_sa.get('probability', 0.5) or 0.5),
            weight=float((_ec or {}).get(
                {'substantive': 'substantive_weight_share',
                 'forecast': 'forecast_weight_share',
                 'market': 'market_weight_share',
                 'official': 'official_weight_share'}.get(
                     (_ec or {}).get('dominant_evidence_kind', ''), 'other_weight_share'), 0.0)),
            display_label=str(_rep_for_sa.get('label', '')),
            original_label=_sa_md.get('original_label'),
            original_probability=(float(_sa_md['original_probability'])
                                   if _sa_md.get('original_probability') else None),
            is_inverted=bool(_sa_md.get('inverted_for_slot') == sid),
            is_slot_view=str(_sa_md.get('is_slot_view', '')).lower() == 'true',
            use_in_graph_simulation=str(_sa_md.get('exclude_from_graph_simulation', '')).lower() != 'true',
            rationale=str(_sa_md.get(f'positive_framing_label__{sid}')
                          or _sa_md.get('slot_relevance_rationale', '') or ''),
        )
        slot_assignments.append(_sa)

        # Phase 6.7.1 (Vince's refinement on GPT's event-centric design):
        # when the REPRESENTATIVE event for this slot was inverted, its
        # displayed label and probability should directly reflect the
        # positive-framed proposition and its effective probability — not a
        # parallel "raw vs effective" split that only the branch group sees.
        # The event's own slider/probability becomes the inverted value; the
        # original extraction (label + raw probability) is preserved in
        # metadata for the "extracted as: ..." display.
        _rep_md = rep.get('metadata') or {}
        if _rep_md.get('inverted_for_slot') == sid:
            _rep_md['original_label'] = rep.get('label')
            # original_probability already set by _register_effective_prob
            rep['label'] = _rep_md.get('positive_framing_label') or rep.get('label')
            rep['probability'] = _rep_md.get('effective_probability', rep.get('probability'))
            rep['metadata'] = _rep_md

        rep_md = rep.get('metadata') or {}
        if str(rep_md.get('synthetic', '')).lower() == 'true':
            synthetic_count += 1
            mapping_methods["synthetic"] += 1
        else:
            organic_count += 1
            if str(rep_md.get('weak_match', '')).lower() == 'true':
                mapping_methods["weak_semantic_match"] += 1
            elif 'mapped_event_ids' in rep or rep_md.get('stage_c_direct'):
                mapping_methods["stage_c_direct"] += 1
            else:
                mapping_methods["semantic_match"] += 1

        # Phase 6.8 metrics (all post-resolution, per GPT guidance point 7)
        if rep_md.get('inverted_for_slot') == sid:
            inverted_count += 1
        if str(rep_md.get('weak_match', '')).lower() == 'true':
            weak_count += 1
        if str(rep_md.get('is_slot_view', '')).lower() == 'true':
            slot_view_count += 1
            reused_source_events += 1
        ev_kind = rep_md.get('event_kind', '')
        if ev_kind in ('forecast_signal', 'market_signal',
                       'expert_assessment', 'public_opinion_signal'):
            meta_signal_count += 1
        _ec_for_sq = slot_evidence_compositions.get(sid, {})
        if _ec_for_sq.get('dominant_evidence_kind') in ('forecast', 'market', 'meta_signal'):
            forecast_dominant_count += 1
        if rep_md.get('horizon_fit') in ('partial', 'beyond', 'unclear'):
            horizon_mismatch_count += 1

        print(f"[v0.12] Slot '{sid}': {len(evs)} event(s) → p={normalized.get(sid,0):.3f} "
              f"(rep={rep['id']})")

    for sid in unmapped_slots:
        print(f"[v0.12] Slot '{sid}': UNMAPPED — insufficient evidence, no synthetic created")

    total_slots = len(slots)
    organic_rate = organic_count / total_slots if total_slots else 0.0
    seed_quality = {
        "primary_slots_total":      total_slots,
        "organic_slots":            organic_count,
        "synthetic_slots":          synthetic_count,
        "unmapped_slots":           len(unmapped_slots),
        "unmapped_slot_ids":        unmapped_slots,
        "organic_rate":             round(organic_rate, 3),
        "synthetic_rate":           round(synthetic_count / total_slots, 3) if total_slots else 0.0,
        "direct_assignment_rate":   round(mapping_methods["stage_c_direct"] / total_slots, 3) if total_slots else 0.0,
        "semantic_fallback_rate":   round(mapping_methods["semantic_match"] / total_slots, 3) if total_slots else 0.0,
        "weak_match_rate":          round(mapping_methods["weak_semantic_match"] / total_slots, 3) if total_slots else 0.0,
        "mapping_methods":          mapping_methods,
        # Phase 6.7/6.8 (GPT guidance, points 7+)
        "inverted_slots":           inverted_count,
        "weak_slots":               weak_count,
        "reused_source_events":     reused_source_events,
        "meta_signal_slots":        meta_signal_count,
        "forecast_dominant_slots":  forecast_dominant_count,
        "slot_view_count":          slot_view_count,
        "horizon_mismatch_count":   horizon_mismatch_count,
        "assignment_collision_count": assignment_collision_count,
    }
    quality_flag = "green" if organic_rate >= 0.80 else ("amber" if organic_rate >= 0.60 else "red")
    seed_quality["quality_flag"] = quality_flag
    print(f"[v0.12] Seed quality: organic_rate={organic_rate:.2f} ({quality_flag}), "
          f"synthetic={synthetic_count}, unmapped={len(unmapped_slots)}, methods={mapping_methods}, "
          f"inverted={inverted_count}, weak={weak_count}, slot_views={slot_view_count}, "
          f"meta_signal={meta_signal_count}, forecast_dominant={forecast_dominant_count}, "
          f"horizon_mismatch={horizon_mismatch_count}")

    _LAST_PRIMARY_QUESTION.update({
        "primary_branch_group_id": bg_id,
        "canonical_outcome_slots": [s.slot_id for s in slots],
        "canonical_slots_full": [
            {"slot_id": s.slot_id, "label": s.label,
             "slot_polarity": s.slot_polarity, "slot_kind": s.slot_kind,
             "description": s.description, "synthetic_allowed": s.synthetic_allowed,
             # Phase 6.6: expose all three probability layers for diagnostics
             "raw_probability": raw_probs.get(s.slot_id),
             "branch_probability": normalized.get(s.slot_id)}
            for s in slots
        ],
        "question_type":           getattr(primary_question, 'question_type', 'binary'),
        "confidence":              0.9,
        "seed_quality":            seed_quality,
    })

    # ── Phase 7.3: Build SlotInfluence records from non-branch events ────────
    # Non-branch events with high slot_affinity or matching Stage C relation
    # data become SlotInfluences — pressures on canonical slots from the
    # wider causal field, without those events becoming branch members.
    slot_influence_records: list = []
    _branch_rep_ids = {slot_rep[sid].get('id') for sid in slot_rep}
    for _ev in event_dicts:
        _ev_id = _ev.get('id', '')
        if _ev_id in _branch_rep_ids:
            continue  # already a branch representative — skip
        if str((_ev.get('metadata') or {}).get('is_slot_view', '')).lower() == 'true':
            continue  # slot-view clone — skip
        _affinities = _ev.get('slot_affinities') or {}
        _ev_md = _ev.get('metadata') or {}
        _ek = _ev_md.get('event_kind', 'substantive_development')
        _ek_w = _EVENT_KIND_WEIGHTS.get(_ek, 1.00)
        for _sid, _aff in _affinities.items():
            if _aff is None: continue
            _aff = float(_aff)
            if _aff < 0.35: continue  # below influence threshold
            # Determine relation for this (event, slot) pair
            _rel_key = f'relation_to_slot__{_sid}'
            _relation = str(_ev_md.get(_rel_key) or _ev_md.get('relation_to_slot') or 'context').lower()
            _conf = _aff * _ek_w
            slot_influence_records.append(SlotInfluence(
                source_event_id  = _ev_id,
                slot_id          = _sid,
                relation_to_slot = _relation,
                influence_weight = _aff * _ek_w,
                confidence       = _conf,
                rationale        = str(_ev_md.get('slot_relevance_rationale', '') or ''),
            ))

    return event_dicts, {"slot_assignments": assignments, "slot_probabilities": normalized,
                          "slot_evidence_compositions": slot_evidence_compositions,
                          "slot_assignment_records": slot_assignments,
                          "slot_influence_records": slot_influence_records,
                          "seed_quality": seed_quality}

# ============================================================
# Phase 5: Stage Q — Primary Question Decomposition
# Runs before Stage C. Defines the canonical answer space from
# the question itself, so Stage C maps events to slots rather
# than inventing the answer space from scratch.
# ============================================================

PROMPT_STAGE_Q = """
You are a question decomposition specialist for a temporal extrapolation engine.
Given the question and time horizon, classify the question type and define the canonical outcome slots that directly and completely answer it.

Every slot must be:
- mutually distinct from the others,
- collectively sufficient to answer the question,
- named with a stable snake_case slot_id,
- labelled clearly with yes/no/partial/status_quo/alternative polarity.

Preserve the possibility of non-occurrence or status quo when logically required.
Do not create unnecessary slots. Do not split one outcome into multiple labels unless the distinction matters for the question.

In each slot description, name the scenario's primary actors and concrete evidence patterns — not generic phrases like "escalation occurs" but actor-specific phrases like "North Korea escalates through ICBM tests or nuclear demonstrations toward neighbors."

For threshold questions only, include: threshold_unit, lower_bound, upper_bound, aliases (natural-language phrases for the bucket, e.g. ["one cut", "25 bps", "modest easing"]).
Optional for all slots: actor_scope (primary actor, e.g. "North Korea"), geographic_scope (domestic_us|international|regional|global), scale_level (national|bilateral|multilateral|global), directionality (continuation|intensification|reversal|breakthrough|decline|uncertain), action_exemplars (list of supporting evidence patterns), negative_examples (list of evidence that does NOT support this slot).

Question: {topic}
Time horizon: {horizon}

QUESTION TYPES:
- binary:           Will X happen? → exactly 2 slots (one yes, one no)
- multi_outcome:    What will X do? → 3-6 slots, each a distinct named outcome
- threshold:        How much/many of X? → one slot per numeric bucket
- open_scenario:    How will X evolve? → 3-6 mutually exclusive scenario paths
- timeline:         When will X happen? → 2-5 time-window slots

Return JSON only — no preamble, no markdown:
{{
  "question_type": "binary|multi_outcome|threshold|open_scenario|timeline",
  "canonical_slots": [
    {{
      "slot_id": "snake_case_id",
      "label": "Clear outcome label",
      "description": "Actor-specific description naming primary actors and evidence patterns.",
      "slot_polarity": "yes|no|partial|status_quo|alternative",
      "slot_kind": "outcome|status_quo|threshold_bucket|timeline_bucket",
      "threshold_unit": "basis_points",
      "lower_bound": 25,
      "upper_bound": 100,
      "aliases": ["one cut", "25 bps"],
      "actor_scope": "North Korea",
      "geographic_scope": "regional",
      "scale_level": "national",
      "directionality": "continuation",
      "action_exemplars": ["North Korea conducts ICBM test", "Kim Jong Un announces arsenal expansion"],
      "negative_examples": ["North Korea signs arms reduction treaty"],
      "evidence_terms": ["missile test", "weapons development", "nuclear capability", "arsenal expansion"]
    }}
  ]
}}

Fields threshold_unit/lower_bound/upper_bound/aliases are only for threshold_bucket slots.
Fields actor_scope, geographic_scope, scale_level, directionality, action_exemplars, negative_examples, and evidence_terms are optional enrichment — include when you can.
evidence_terms: vocabulary that sources would actually use when reporting evidence for this slot (e.g. for a "continuation" slot: ["maintained", "continued testing", "no major change", "steady"]).
""".strip()



def _slot_synthetic_allowed(slot_polarity: str, slot_kind: str, question_type: str) -> bool:
    """
    Phase 6.5 (GPT guidance): synthetic placeholders are allowed only for
    logically-required complement slots — not as a general fallback.
      - Binary "no" complement: allowed (the null outcome is logically required)
      - Status quo / no-change slots: allowed (sources often under-discuss continuity)
      - Timeline "not within horizon" complements: allowed
      - Open-scenario generated slots: NOT allowed by default
        (if the engine invented the scenario path, it should not also invent
         evidence for it)
    """
    if slot_kind == "status_quo":
        return True
    if slot_polarity in ("no",) and question_type == "binary":
        return True
    if slot_kind == "timeline_bucket" and slot_polarity in ("no", "alternative"):
        return True
    return False

def run_stage_q(topic: str, horizon_label: str) -> 'PrimaryQuestion':
    """
    Stage Q: Decompose the primary question into canonical outcome slots.
    Uses Haiku for speed — this is a classification task, not generation.
    Returns a PrimaryQuestion with canonical_slots.
    """
    print("[v0.12] Stage Q: Primary question decomposition...")
    if not LIVE_MODE:
        # Stub: return a simple binary
        return PrimaryQuestion(
            text=topic, question_type="binary", horizon_label=horizon_label,
            canonical_slots=[
                OutcomeSlot("yes_outcome", "Yes, this occurs",
                            "The described event occurs within the horizon",
                            slot_polarity="yes", slot_kind="outcome",
                            synthetic_allowed=_slot_synthetic_allowed("yes","outcome","binary")),
                OutcomeSlot("no_outcome", "This does not occur",
                            "The described event does not occur within the horizon",
                            slot_polarity="no", slot_kind="outcome",
                            synthetic_allowed=_slot_synthetic_allowed("no","outcome","binary")),
            ]
        )

    result = call_anthropic_api(
        "You are a question decomposition specialist. Return only valid JSON.",
        PROMPT_STAGE_Q.format(topic=topic, horizon=horizon_label or "the stated horizon"),
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        timeout=45,
    )

    if not isinstance(result, dict) or not result.get("canonical_slots"):
        print("[v0.12] Stage Q fallback: defaulting to binary slots")
        return PrimaryQuestion(
            text=topic, question_type="binary", horizon_label=horizon_label,
            canonical_slots=[
                OutcomeSlot("yes_outcome", topic[:60], "", slot_polarity="yes",
                            synthetic_allowed=False),
                OutcomeSlot("no_outcome",  "Does not occur", "", slot_polarity="no",
                            slot_kind="outcome",
                            synthetic_allowed=_slot_synthetic_allowed("no","outcome","binary")),
            ]
        )

    q_type = str(result.get("question_type", "binary"))
    slots = []
    for s in result.get("canonical_slots", []):
        if not isinstance(s, dict): continue
        s_polarity = str(s.get("slot_polarity", "yes"))
        s_kind     = str(s.get("slot_kind",     "outcome"))
        slots.append(OutcomeSlot(
            slot_id       = str(s.get("slot_id",  "slot"))[:40],
            label         = str(s.get("label",    "Outcome")),
            description   = str(s.get("description", "")),
            slot_polarity = s_polarity,
            slot_kind     = s_kind,
            synthetic_allowed = _slot_synthetic_allowed(s_polarity, s_kind, q_type),
        ))

    # Safety: need at least 2 slots
    if len(slots) < 2:
        slots.append(OutcomeSlot("no_outcome", "Does not occur within horizon",
                                  "", slot_polarity="no", slot_kind="outcome",
                                  synthetic_allowed=_slot_synthetic_allowed("no","outcome",q_type)))
    pq = PrimaryQuestion(
        text=topic, normalized_question=topic.lower().strip(),
        question_type=q_type, horizon_label=horizon_label,
        canonical_slots=slots,
    )
    print(f"[v0.12] Stage Q: {q_type} question, {len(slots)} canonical slots: "
          f"{[s.slot_id for s in slots]}")
    return pq

PROMPT_STAGE_C_STRUCTURE = TEMPORAL_OBSERVER_PROTOCOL + """

---

Act as a Temporal Observer preserving causality and self-consistency.

Map candidate events to canonical outcome slots only when they genuinely bear on that slot. For each mapping, state whether the event supports, opposes, contextualizes, or merely signals the slot.

An event that argues against a slot is still relevant to that slot. Mark it as "opposes"; do not reject it solely because it is framed negatively.

CRITICAL DISTINCTION — causal_context vs. substantive_development:

Use "causal_context" ONLY when the event describes a mechanism, driver, pressure, or condition that INFLUENCES which outcome may occur but is not itself the state of the world the outcome represents.

Use "substantive_development" when the event describes a state of the world that the outcome itself REPRESENTS.

Persistent states — ongoing sanctions regimes, diplomatic stalemates, frozen conflicts, active regulatory frameworks, continuing market conditions, diplomatic paralysis, or active military postures — may themselves BE the status-quo outcome rather than merely influencing it. Do NOT classify persistent background-state events as "causal_context" if they directly represent a canonical slot such as: status quo, continuation, divergence, fragmentation, stalemate, non-resolution, or paralysis.

Examples:
- "Sanctions erosion is ongoing" → this IS the status-quo condition, not a cause → substantive_development
- "Regulatory fragmentation persists" → this IS divergence → substantive_development
- "Domestic political opposition makes agreement harder" → this INFLUENCES outcomes → causal_context

CRITICAL — Actor and scope verification (GPT Phase 7 guidance):
Before mapping an event to a slot, verify that the event's subject actors and geographic scope match the slot's defined actor scope and scale level. A US state-level regulatory event is NOT direct evidence for a US-EU-China international slot — it may be contextual or causal evidence at most, and should be marked as causal_context or evidence_context, not substantive_development for that slot. A sub-national event for a national slot should be carefully examined.

Also verify directionality: a slot defined as status_quo or continuation expects evidence of persistence or maintenance of existing conditions. An event describing capability growth, escalation, or intensification is likely a DIFFERENT slot (intensification, acceleration, escalation) rather than a status_quo slot — do not force such events into status_quo unless the status quo is explicitly defined as "ongoing intensification."

If the slot has actor_scope defined (e.g., "North Korea" or "US-EU-China"), your mapped event should involve those actors. If it does not, explain the connection explicitly in slot_relevance_rationale, or reclassify the event as causal_context.

For every slot mapping, write "slot_relevance_rationale" using explicit terms from the slot's label, slot_id, or description. The engine checks your rationale for vocabulary overlap with the slot to confirm you are reasoning about this exact slot. A good rationale answers:
- What part of this event bears on this slot?
- Does it support, oppose, contextualize, or signal the slot?
- Why is this event about the slot's outcome rather than merely about the broader topic?
- If the slot has action_exemplars or evidence_terms, mention which pattern this event resembles.

ACTION EXEMPLARS: Before deciding that no event maps to a slot, check the slot's action_exemplars, evidence_terms, and aliases. If the event resembles any of those patterns, mention the matching exemplar in your slot_relevance_rationale. This is especially important for slots with abstract labels that may not use the same vocabulary as source reporting.

REQUIRED FIELDS — when best_event_id is not null, all of the following are required:
  relation_to_slot, relation_confidence, event_kind, slot_relevance_rationale.
  If relation_to_slot is "supports" or "opposes", positive_framing_label is also required.
  If relation_to_slot is "opposes", direct_negation (boolean) is also required.
  Do not omit these fields. The engine uses them for acceptance, inversion, and display.

RELATION DEFINITIONS — use these precisely:
  "supports"           — the event directly supports this slot's outcome occurring.
  "opposes"            — the event explicitly negates this slot's outcome: states it will not occur,
                         is ruled out, fails, collapses, or is impossible. ONLY use for direct negation.
                         When using "opposes", set direct_negation: true and write a positive_framing_label.
  "alternative_pathway"— the event supports a neighboring, partial, substitute, or adjacent outcome
                         that belongs to another slot. Route these to the appropriate slot instead.
                         Do NOT invert alternative_pathway events. Do NOT use "opposes" for them.
  "context"            — the event shapes the field but is not itself evidence for or against this slot.
  "meta_signal"        — the event reports forecasts, market expectations, polling, or expert estimates.

INVERSION SAFETY RULE:
  "opposes" should only fire when the event directly states this outcome will NOT occur.
  Examples of CORRECT opposes: "Harmonization will not occur", "Agreement is ruled out", "Convergence collapses."
  Examples of WRONG opposes (use alternative_pathway instead):
    "US and EU align without China" → this describes bilateral_clusters, not a negation of trilateral harmonization.
    "100–200 bps cut" for ">200 bps" slot → this is an adjacent threshold bucket, not a negation.
    "North Korea maintains current posture" for escalation slot → this supports status_quo, not opposes escalation.

---

Given the full candidate event set below (with scores and Stage A hints),
map events to the canonical outcome slots and infer the full relational structure.

Candidate events:
{events_with_hints}

Topic: {topic}

CANONICAL OUTCOME SLOTS (defined by the primary question — do NOT change these):
{canonical_slots_text}

STEP 0 — PRIMARY BRANCH GROUP from canonical slots (mandatory first):
The canonical outcome slots above define the complete answer space for the primary question.
For each slot, identify candidate events from the list below that bear on it (events may
have slot_affinities already indicating this from Stage A — use those as a starting point,
then verify against the event's full label/description).

CRITICAL — "candidate" means evidence ABOUT the slot's likelihood, not evidence FOR a high
likelihood. An event that argues this outcome will NOT happen, or is unlikely, is still a
valid candidate for that slot — it is direct evidence that the slot's probability should be
LOW, which is itself informative and should appear in the branch group rather than be
omitted. Do NOT reject a candidate merely because it argues against the outcome occurring.
Only reject a candidate if it provides NO information about this specific slot's likelihood
at all (i.e. it is about a different question entirely).
Example: if the slot is "Trilateral AI standards converge" and an event says "Trilateral
harmonization will not materialize due to regulatory fragmentation," that event IS a valid
candidate for the slot — map it with an appropriately low probability, do not return
best_event_id: null.

For each slot, return:
  - candidate_event_ids: ranked list of event_ids that plausibly represent this slot
    (best match first). Empty list if nothing plausibly fits.
  - best_event_id: the single best candidate, or null if no candidate clears a reasonable
    confidence bar.
  - match_confidence: 0-1, your confidence that best_event_id genuinely represents this slot.
    >= 0.55 = strong match. 0.40-0.55 = weak but usable match. < 0.40 = treat as no match.
  - no_organic_match_reason: ONLY if best_event_id is null — a short note on why
    (e.g. "No extracted event addresses this outcome; sources did not discuss it").

  When best_event_id is set, ALSO return:
  - relation_to_slot: how best_event_id relates to this slot's proposition —
      "supports" = event describes this outcome occurring / being true
      "opposes" = event describes the NEGATION of this outcome (e.g. "X will not
                  materialize" for a slot about X occurring) — STILL a valid,
                  informative mapping; the engine will invert its probability
      "context" = event provides background/causal context but doesn't directly
                  state whether this outcome occurs
      "meta_signal" = event is ABOUT forecasts/predictions/markets/polls regarding
                  this outcome, not the outcome itself
      "irrelevant" = should not have been proposed as a candidate at all
  - relation_confidence: 0-1, your confidence in the relation_to_slot classification
    (this can be HIGH even when match_confidence reflects lexical uncertainty —
    e.g. you can be very confident an event OPPOSES a slot even if its wording
    shares few words with the slot's label).
  - slot_relevance_rationale: one sentence explaining WHY this event bears on this
    slot's proposition — be specific about what the event says and how it maps to
    the slot, so the engine can verify your reasoning even when lexical overlap is low.
  - positive_framing_label: REQUIRED whenever relation_to_slot is "supports" or
    "opposes" (optional/omit for context/meta_signal/irrelevant). A short
    rephrasing of the event's OWN proposition in the SAME polarity as this slot
    (i.e., if the event is "X will not occur" and the slot is about X occurring,
    give "X occurs" / "X materializes" — a positive, non-negated framing of what
    the event is actually evidence ABOUT, NOT the slot's label). This is the
    PRIMARY label shown to users for this event — used so they see an
    event-centric framing like "X occurs — 22%" rather than the event's raw
    label paired with a confusing probability, or the slot's abstract label
    standing in for the event.
  - event_kind: classify the event itself —
      "substantive_development" = a real-world event/development or persistent condition
                  that the outcome itself represents (not merely influences it)
      "official_action" = a government/institutional action, policy decision, or
                  enacted/implemented policy
      "forecast_signal" = an analyst/expert prediction about the future
      "market_signal" = a prediction market, betting odds, or similar
      "expert_assessment" = a named expert's stated opinion/assessment
      "public_opinion_signal" = polling or public sentiment data
      "causal_context" = a mechanism, driver, pressure, or condition that INFLUENCES
                  which outcome occurs but is NOT itself the state of the world
                  the outcome represents; do NOT use for persistent states
                  that directly represent status-quo, continuation, divergence,
                  fragmentation, or non-resolution outcomes
      "evidence_context" = source/documentation about the historical record

Do NOT decide whether to create a synthetic placeholder — that is the engine's decision,
not yours. Your job is honest mapping and confidence scoring, not filling gaps.
Do NOT replace or modify the canonical slots — only map events to them.

Tasks:

1. BRANCH GROUPS: mutually exclusive alternative outcomes of the same resolution question.

   FUNDAMENTAL RULE: A branch group represents a field of mutually exclusive possible futures
   answering one specific question. It is NOT a theme or topic cluster.

   CRITICAL NAMING RULES:
   - branch_group key MUST be descriptive snake_case with NO numeric prefix.
   - GOOD: "iran_nuclear_negotiation_outcome", "us_eu_tariff_decision", "ceasefire_status"
   - BAD: "BG_001", "BG1", "BG3_something", "group_1"
   - Event IDs must ALWAYS use EVT_### format (EVT_001, EVT_002, etc.)

   CRITICAL STRUCTURE RULES:
   - ALWAYS ensure the user's primary scenario question is answered by a multi-member branch group.
   - Never place mutually exclusive outcomes in separate single-member branch groups.
   - Do NOT create a branch group with only one member. Prefer branch_group: null over single-member groups.
   - If two events are exclusive (one prevents the other), they MUST share a branch group.
   - Minimum 2 members per branch group. ALWAYS include a "no significant change / status quo maintained" outcome.
     This is mandatory even when the question asks "will X happen or Y happen" — the status-quo option
     ensures the field reflects all three possibilities (X, Y, or neither). Without it, the branch group
     implies the scenario WILL escalate when it might not.
     BAD: Only "Turkey reduces NATO commitments" and "Turkey leaves NATO" (missing "Turkey maintains commitments")
     GOOD: "Turkey maintains current commitments" + "Turkey reduces commitments" + "Turkey leaves NATO"
   - NEVER place events with temporal_status "resolved" inside a branch group.
     Resolved events are past facts — they belong as standalone events with causal edges.
   - Events with temporal_status "active" MAY be placed in a branch group ONLY as the
     status-quo / no-change member (representing the current ongoing situation continuing).
     BAD: putting a resolved event like "Law passed in 2023" inside a branch group.
     GOOD: "Inter-Korean stalemate continues (active)" as the status-quo member alongside
           "Diplomatic dialogue resumes (unresolved)" as the change member.

   CRITICAL LABEL RULES — branch_label must describe EXACTLY ONE outcome:
   - NEVER use "OR", "and/or", "vs", "versus", "either", "alternative", or slash-separated outcomes.
   - If source language contains "X or Y," create TWO separate branch members: one for X, one for Y.
   - Each member must be mutually exclusive with every other member in its group.
   - BAD:  "Ceasefire holds OR war resumes"
   - GOOD: "Ceasefire holds through horizon" + "War resumes before horizon"
   - BAD:  "Deal reached / conflict breakdown"
   - GOOD: "Comprehensive nuclear deal reached" + "Conflict breakdown without deal"
   - BAD:  Creating BG_001 with only "Tariffs imposed"
   - GOOD: "us_eu_tariff_decision" with "Higher tariffs imposed" + "Deal reached avoiding tariffs"

2. CAUSAL EDGES: directed relationships.
   relation_type: causal | reinforcing | inhibitory | exclusive | enabling | feedback
   causal_weight 0-1, uncertainty 0-1, feedback_strength 0-1.

Return JSON only (no fences, no prose):
{{
  "primary_branch_group": {{
    "branch_group_id": "primary_outcome",
    "members": [
      {{
        "slot_id": "yes_outcome",
        "label": "The outcome label shown to the observer",
        "slot_polarity": "yes",
        "slot_kind": "outcome",
        "outcome_role": "primary_outcome",
        "candidate_event_ids": ["EVT_003", "EVT_007"],
        "best_event_id": "EVT_003",
        "match_confidence": 0.82,
        "relation_to_slot": "supports",
        "relation_confidence": 0.82,
        "slot_relevance_rationale": "Directly describes this outcome occurring within the horizon.",
        "positive_framing_label": "The outcome occurring, phrased to match this slot's polarity",
        "event_kind": "substantive_development"
      }},
      {{
        "slot_id": "no_outcome",
        "label": "Does not occur before horizon",
        "slot_polarity": "no",
        "slot_kind": "outcome",
        "outcome_role": "primary_outcome",
        "candidate_event_ids": [],
        "best_event_id": null,
        "match_confidence": 0.0,
        "no_organic_match_reason": "No extracted event describes continuity or no-change."
      }}
    ]
  }},
  "secondary_branch_groups": [
    {{"branch_group": "descriptive_snake_case_key", "resolution_question": "...",
      "members": [{{"event_id": "EVT_001", "branch_label": "Single coherent outcome label"}}],
      "mutual_exclusivity_confidence": 0.9}}
  ],
  "causal_edges": [
    {{"source_id": "EVT_001", "target_id": "EVT_002", "relation_type": "causal",
      "causal_weight": 0.6, "uncertainty": 0.3, "feedback_strength": 0.0}}
  ]
}}

primary_branch_group: one member per canonical slot, with candidate_event_ids/best_event_id/
match_confidence as described in STEP 0. The ENGINE — not you — decides whether an unmapped
slot becomes a synthetic placeholder, gets a logical complement, or is left as an audit gap.
secondary_branch_groups: additional genuine branch dynamics beyond the primary question.
outcome_role for primary branch members: "primary_outcome" or "status_quo_outcome".
Events with outcome_role="causal_context" or "evidence_context" must retain independent
probabilities and must NOT be forced to sum to 1.0 with other events.

MANDATORY PRIMARY BRANCH GROUP RULE:
You must return exactly one member for every canonical slot provided in CANONICAL OUTCOME SLOTS.
Do NOT omit any canonical slot. Do NOT rename slot_id values.
Report your honest mapping confidence — do not force a match just to avoid an empty
candidate_event_ids list. A low match_confidence or empty candidate list is valuable
information, not a failure.
Do NOT invent new primary slots. Slot_id is the identity — branch labels are display text only.

If these slots are marked [LOCKED] (temporal re-observation), do not invent, remove, or rename
any primary outcome slot. Map current evidence back onto the locked slots only.
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

def generate_search_queries(topic: str, observer_basis: dict, canonical_slots=None,
                              horizon: "HorizonConfig" = None) -> List[dict]:
    """
    Stage 0: Query planning — hybrid fan-out.
    One broad field scan + narrow measurement-basis projections.
    Stage A observes the result; Stage A does NOT plan the initial queries.

    Phase 6.5 (GPT guidance, Pass 2): since Stage Q now runs before Stage 0,
    canonical_slots are available here. For slots where synthetic_allowed=False
    (i.e. slots that MUST find organic evidence — open-scenario outcome slots),
    add one targeted query per slot so the search field is slot-aware from
    the start. Capped to avoid excessive query fan-out.

    Phase 6.8 (GPT guidance, point 2): horizon is now a first-class input.
    Queries include the target year so search engines surface results framed
    within the actual scenario window, not just the current calendar year.

    Phase 7 (Temporal Observer Protocol — Stage 0 insert):
    Seek source diversity across direct evidence, causal drivers, institutional
    context, opposing evidence, and status-quo evidence. Do not search only for
    dramatic outcomes. For every major outcome slot, seek organic evidence that
    could support or oppose it.
    """
    import datetime as _dt
    _today_year = _dt.date.today().year
    _horizon_year = horizon.target_date.year if (horizon and hasattr(horizon, 'target_date')) else _today_year + 1
    _horizon_str = f"through {horizon.target_date_str}" if horizon else ""

    queries = [
        {"query": f"{topic} latest developments {_today_year}",
         "purpose": "field_scan", "source_type": "article", "max_results": 5},
        {"query": f"{topic} outlook forecast {_horizon_year}",
         "purpose": "horizon_forecast", "source_type": "forecast", "max_results": 4},
        {"query": f"{topic} risk assessment {_today_year} {_horizon_year}",
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

    # Phase 6.5: slot-aware queries for slots requiring organic evidence
    if canonical_slots:
        slot_queries_added = 0
        for slot in canonical_slots:
            if slot_queries_added >= 3:
                break  # cap to avoid excessive fan-out
            if getattr(slot, 'synthetic_allowed', False):
                continue  # this slot can fall back to a placeholder; lower priority
            # Phase 6.8: include horizon boundary in slot-targeted queries
            _slot_q_horizon = f" {_horizon_str}" if _horizon_str else ""
            queries.append({
                "query": f"{topic} {slot.label}{_slot_q_horizon}",
                "purpose": f"slot_evidence_{slot.slot_id}",
                "source_type": "article",
                "max_results": 3,
            })
            slot_queries_added += 1

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
    Prediction market and gambling sites are filtered out before fetching.
    """
    # Filter prediction market / gambling domains — their probability language
    # looks like event data but is actually crowd-sourced betting, not analysis.
    filtered = []
    for sr in search_results:
        domain = sr.url.lower()
        if any(blocked in domain for blocked in PREDICTION_MARKET_DOMAINS):
            print(f"[SOURCE_FILTER] Skipping prediction market URL: {sr.url[:60]}")
            continue
        filtered.append(sr)
    search_results = filtered

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


def _clean_json_text(text: str) -> str:
    """Strip control characters and markdown fences from AI JSON responses."""
    # Remove control chars (keep tab=9, newline=10, CR=13)
    text = "".join(c for c in text if ord(c) >= 32 or ord(c) in (9, 10, 13))
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return t


def _repair_truncated_json(text: str):
    """
    Salvage a JSON response truncated by max_tokens.
    Finds the last complete JSON object and closes any open structures.
    Returns a parsed dict/list or raises ValueError.
    """
    import json

    t = _clean_json_text(text)

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
        try:
            result = _repair_truncated_json(text)
        except Exception:
            return {}   # malformed beyond repair — caller handles empty result
    if isinstance(result, str):
        return {}       # bare string — not a valid API response structure
    return result


def extract_candidate_events_holistic(topic, source_summaries, observer_basis,
                                       temporal_context="", canonical_slots=None):
    """
    Stage A: holistic extraction + evidence gap detection.
    Returns (event_dicts, follow_up_queries, missing_evidence).

    Phase 6.5 (GPT guidance, Pass 2): canonical_slots from Stage Q (which now
    runs BEFORE Stage A) are passed in so extraction is slot-aware from the
    start. Each event includes a slot_affinities field — which canonical
    slot(s) it bears on and how strongly — making Stage C's mapping job
    much easier downstream.
    """
    if not LIVE_MODE:
        return [], [], []
    basis_str = "\n".join(f"  {k}: {v}" for k,v in observer_basis.items())
    # Prepend temporal context so the model knows today's date
    # Phase 7 (GPT guidance): include threshold slot alias hints in temporal context
    # so Stage A knows to look for natural-language equivalents of numeric thresholds
    _threshold_aliases_text = ""
    if canonical_slots:
        threshold_slots = [s for s in canonical_slots
                           if getattr(s, 'slot_kind', '') == 'threshold_bucket'
                           and getattr(s, 'aliases', None)]
        if threshold_slots:
            alias_lines = []
            for ts in threshold_slots:
                alias_lines.append(
                    f"  {ts.slot_id} ({ts.label}): aliases = {ts.aliases}"
                )
            _threshold_aliases_text = (
                "THRESHOLD SLOT ALIASES (natural-language phrases that map to numeric bucket slots):\n"
                + "\n".join(alias_lines) + "\n"
                "When extracting events, look for these phrases as evidence for the corresponding slot.\n\n"
            )

    tc_prefix = (f"TEMPORAL CONTEXT: {temporal_context}\n"
                 + _threshold_aliases_text
                 + "Classify temporal_status: resolved=already occurred, "
                 "active=ongoing now, unresolved=may occur within horizon, "
                 "beyond_horizon=after horizon end, counterfactual=alternative path.\n"
                 "Phase 6.8: For each event, also provide:\n"
                 "  expected_window_label: a brief description of when this event "
                 "is expected to resolve (e.g. '2025', '2026-2027', 'next 5 years', "
                 "'before 2031')\n"
                 "  horizon_fit: classify based on COVERAGE of the full scenario "
                 "window, not just whether the event occurs before the horizon end:\n"
                 "    'inside' = the event directly addresses whether the scenario "
                 "question resolves across the FULL horizon window (e.g. a statement "
                 "about the full 5-year period, or an event with no tight time "
                 "constraint that remains relevant throughout)\n"
                 "    'partial' = the event is relevant BUT only covers a SHORTER "
                 "window than the full scenario (e.g. a 1-year or 2-year forecast "
                 "framed as '2026-2027' for a 5-year scenario — still valid evidence, "
                 "but does not address what happens in the remaining years)\n"
                 "    'beyond' = event is relevant but expected to resolve AFTER the "
                 "horizon end\n"
                 "    'unclear' = timing is genuinely ambiguous\n"
                 "  Key rule: if the event's label contains a year-range shorter than "
                 "the scenario window (e.g. '2026-2027' for a horizon ending 2031), "
                 "classify as 'partial', not 'inside', even if that year-range falls "
                 "within the horizon. Include partial-coverage events rather than "
                 "omitting them — partial evidence is still evidence.\n\n"
                 if temporal_context else "")

    # Phase 6.5: build canonical slots context for slot-aware extraction
    slots_prefix = ""
    if canonical_slots:
        slot_lines = "\n".join(
            f"  - slot_id={s.slot_id}: {s.label}"
            + (f" ({s.description})" if getattr(s, 'description', '') else "")
            for s in canonical_slots
        )
        slots_prefix = (
            "CANONICAL OUTCOME SLOTS (the answer space this scenario must address):\n"
            f"{slot_lines}\n\n"
            "For each extracted event, include a slot_affinities field: a dict mapping "
            "slot_id -> affinity (0-1) for any canonical slot this event provides evidence "
            "for or against. An event may have affinities to zero, one, or multiple slots. "
            "Prioritize finding at least one organic event with meaningful affinity "
            "(>= 0.4) for EACH canonical slot above — this is the single most important "
            "extraction goal. If you cannot find direct evidence for a slot, extract the "
            "closest related event and note its affinity honestly (it may be lower, e.g. "
            "0.3-0.4) rather than omitting it.\n\n"
        )

    result = call_anthropic_api(
        "You are a UOT temporal extrapolation engine. Return only valid JSON.",
        slots_prefix + tc_prefix + PROMPT_HOLISTIC_EXTRACTION.format(
            topic=topic, observer_basis=basis_str, source_summaries=source_summaries),
        model="claude-sonnet-4-6",
        max_tokens=5500,   # slightly increased to accommodate slot_affinities
        timeout=90
    )
    if isinstance(result, list):
        return result, [], []
    if not isinstance(result, dict):
        return [], [], []
    raw_events = result.get("events", [])
    events     = [e for e in raw_events if isinstance(e, dict)] if isinstance(raw_events, list) else []
    return events, result.get("follow_up_search_queries", []), result.get("missing_evidence", [])


def score_event_fields_specialized(event_candidates, source_summaries):
    """
    Stage B: single combined scoring call (replaces three separate calls).
    Uses claude-haiku for speed — 3 calls reduced to 1.
    """
    if not LIVE_MODE:
        return event_candidates
    summary = "\n".join(
        f"- {e['id']}: {e['label']} | {e.get('description','')[:80]}"
        for e in event_candidates if isinstance(e, dict)
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


# ── Phase 4: PrimaryQuestion — module-level storage for current seeding run ──
# Cleared at the start of every seeding call so stale values never bleed between runs.
_LAST_PRIMARY_QUESTION: dict = {}

def clear_primary_question_state() -> None:
    """Call at the very start of each seeding pipeline to prevent state bleed."""
    global _LAST_PRIMARY_QUESTION
    _LAST_PRIMARY_QUESTION = {}

# ══════════════════════════════════════════════════════════════════════════════
# ID Canonicalization and Branch Validation (Phase 3.6)
# Per GPT guidance: enforce EVT_### event IDs and descriptive snake_case branch
# group names; validate branch structure for single-member groups, exclusive pairs
# not sharing a branch group, and OR-label violations.
# ══════════════════════════════════════════════════════════════════════════════

import re as _re

def _to_snake(text: str) -> str:
    """Convert arbitrary text to snake_case."""
    text = _re.sub('[^a-zA-Z0-9 _]', '', text.lower())
    text = _re.sub(' +', '_', text.strip())
    text = _re.sub(r'_+', '_', text)
    return text[:60]  # cap length


def canonicalize_extraction(event_dicts: list, stage_c_edges: list = None) -> list:
    """
    Normalize event IDs to EVT_### and branch group names to descriptive
    snake_case. Preserves originals in metadata for audit trail.
    Also remaps source_id/target_id in stage_c_edges to match new IDs.
    """
    # Build an ID mapping: old_id → EVT_### (reassign only non-EVT_### IDs)
    id_map = {}
    counter = [0]
    for ev in event_dicts:
        if not isinstance(ev, dict): continue
        old_id = str(ev.get('id', '') or '')
        if _re.match(r'^EVT_\d{3,}$', old_id):
            id_map[old_id] = old_id   # already canonical
        else:
            counter[0] += 1
            new_id = f"EVT_{counter[0]:03d}"
            id_map[old_id] = new_id

    # Apply ID remapping to events
    for ev in event_dicts:
        if not isinstance(ev, dict): continue
        old_id = str(ev.get('id', ''))
        ev.setdefault('metadata', {})['original_event_id'] = old_id
        ev['id'] = id_map.get(old_id, old_id)
        for fld in ('branch_group',):
            pass  # branch_group is a name, not an event ID

    # Apply ID remapping to edges so graph connections survive canonicalization
    if stage_c_edges:
        for edge in stage_c_edges:
            if isinstance(edge, dict):
                if edge.get('source_id') in id_map:
                    edge['source_id'] = id_map[edge['source_id']]
                if edge.get('target_id') in id_map:
                    edge['target_id'] = id_map[edge['target_id']]

    # Canonicalize branch group names
    bg_map = {}
    for ev in event_dicts:
        if not isinstance(ev, dict): continue
        bg = ev.get('branch_group')
        if not bg: continue
        if bg in bg_map: continue
        if _re.match(r'^[a-z][a-z0-9_]{2,59}$', str(bg)) and not _re.match(r'^bg_?\d', str(bg).lower()):
            bg_map[bg] = bg   # already canonical
        else:
            # Try to derive a descriptive name from the branch group or its members
            canonical = _to_snake(
                _re.sub(r'^(BG\d*_?|bg_?\d+_?)', '', str(bg)).strip('_') or 'unresolved_outcome'
            )
            if not canonical or canonical == 'unresolved_outcome':
                canonical = 'unresolved_outcome'
            bg_map[bg] = canonical

    for ev in event_dicts:
        if not isinstance(ev, dict): continue
        bg = ev.get('branch_group')
        if bg and bg in bg_map:
            ev.setdefault('metadata', {})['original_branch_group'] = bg
            ev['branch_group'] = bg_map[bg]

    return event_dicts


def validate_branch_structure(event_dicts: list, primary_branch_group_id: str = None) -> tuple:
    """
    Validate branch group structure per Phase 3.6 rules. Returns (event_dicts, flags).
    Flags are strings describing problems found; does not auto-repair.

    Phase 6.5 (GPT guidance): the PRIMARY branch group is exempt from the
    SINGLE_MEMBER_GROUP check. normalize_and_aggregate_primary_slots may
    deliberately leave a slot unmapped (insufficient evidence) rather than
    inventing a synthetic, which can legitimately reduce the primary group
    to fewer members — including one. A single-member primary group is a
    valid "this scenario field currently has one well-evidenced outcome and
    N unmapped possibilities" state, not a Stage C error to clean up.
    Secondary/auxiliary branch groups still get the original treatment.
    """
    flags = []
    groups: dict = {}
    for ev in event_dicts:
        if not isinstance(ev, dict): continue
        bg = ev.get('branch_group')
        if bg:
            groups.setdefault(bg, []).append(ev)

    for bg_key, members in groups.items():
        # Single-member branch group — exempt the primary branch group
        if len(members) == 1 and bg_key != primary_branch_group_id:
            flags.append(
                f"SINGLE_MEMBER_GROUP: '{bg_key}' has only one member "
                f"('{members[0].get('label', '?')}') — branch_group set to null."
            )
            members[0]['_original_branch_group'] = bg_key
            members[0]['branch_group'] = None
            members[0]['branch_label'] = None

        # Resolved events in branch groups — remove them, they're past facts not future possibilities
        # NOTE: 'active' events (ongoing situations) ARE valid as status-quo branch members
        # and must NOT be removed. Only 'resolved' events (already happened) are excluded.
        #
        # Phase 6.6.6 (same rationale as the SINGLE_MEMBER_GROUP exemption above):
        # the PRIMARY branch group is exempt. A primary-group member's branch_group/
        # branch_label/canonical_slot_id are assigned by normalize_and_aggregate_
        # primary_slots based on canonical SLOT mapping — that is the authoritative
        # signal for primary-group membership. Status-quo-type slots ("Continuation
        # of current trajectory") frequently get temporal_status="resolved" from
        # Stage A, since they describe an already-established/ongoing condition
        # rather than a future event. Stripping such a member here silently removes
        # a correctly-mapped canonical slot from the branch group post-hoc, with no
        # trace other than this flag — exactly the "Step 6 computed it correctly but
        # it never appears in the final output" failure mode.
        if bg_key != primary_branch_group_id:
            for ev in members[:]:
                status = ev.get('temporal_status', 'unresolved')
                if status == 'resolved':
                    flags.append(
                        f"RESOLVED_IN_GROUP: '{bg_key}' contains '{ev.get('label','?')}' "
                        f"with status='resolved' — removed (past fact, not a future possibility). "
                        f"Place as standalone event with causal edges to group members."
                    )
                    ev['_original_branch_group'] = bg_key
                    ev['branch_group'] = None
                    ev['branch_label'] = None

        # OR-label violation
        or_pattern = _re.compile(r'or|and/or|\/|vs\.?|versus|either', _re.IGNORECASE)
        for ev in members:
            lbl = ev.get('branch_label', '') or ''
            if or_pattern.search(lbl):
                flags.append(
                    f"OR_LABEL: '{bg_key}' member '{lbl}' contains ambiguous language. "
                    f"Consider splitting into two distinct outcomes."
                )

    # ── Second pass: groups that became single-member after active/resolved removal ─
    post_groups: dict = {}
    for ev in event_dicts:
        if isinstance(ev, dict) and ev.get('branch_group'):
            post_groups.setdefault(ev['branch_group'], []).append(ev)
    for _bg, _rem in post_groups.items():
        if len(_rem) == 1 and _bg != primary_branch_group_id:
            flags.append(f"BECAME_SINGLE: '{_bg}' reduced to 1 member — nulled.")
            _rem[0]['branch_group'] = None
            _rem[0]['branch_label'] = None

    # Note: Phase 5 canonical outcome slots now ensure status-quo / no-answer presence.
    # The third pass (EVT_SQ creation) and DEDUP_SQ pass are no longer needed.

    # Detect exclusive-edge pairs not in the same branch group (best-effort)
    ev_by_id = {ev.get('id'): ev for ev in event_dicts if isinstance(ev, dict)}
    # We can't check edges here (they're not yet attached), but we can check
    # pairs of events with near-opposite labels in different groups
    # (skip for now — handled in Stage C prompt)

    return event_dicts, flags


# ensure_primary_branch_group removed in Phase 5 — Stage Q canonical slots handle this.

def build_seeded_events_from_dicts(event_dicts):
    """Converts raw extractor output dicts into SeededEvent objects."""
    seeded = []
    for d in (x for x in (event_dicts or []) if isinstance(x, dict)):
        _nd_raw = d.get("extraction_notes"); nd = _nd_raw if isinstance(_nd_raw, dict) else {}
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
        ) for s in d.get("sources",[]) if isinstance(s, dict)]
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
            outcome_role=d.get("outcome_role", "causal_context"),
            causal_candidates=d.get("causal_candidates",[]),
            downstream_impact=float(d.get("downstream_impact",0.5)),
            disruption_score=float(d.get("disruption_score",0.5)),
            novelty=float(d.get("novelty",0.5)),
            causal_support=float(d.get("causal_support",0.5)),
            extraction_notes=diag,
            # Phase 6.6/6.7 carry-through (see SeededEvent field comments)
            canonical_slot_id=d.get("canonical_slot_id"),
            branch_probability=d.get("branch_probability"),
            slot_raw_probability=d.get("slot_raw_probability"),
            seed_metadata=d.get("metadata", {}) or {},
        ))
    return seeded


def infer_branch_groups_and_causal_candidates_specialized(
    event_candidates: list,
    source_summaries: str,
    topic: str,
    **kwargs
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
        # canonical_slots_text is passed in from the pipeline via primary_question parameter
        _pq_param = kwargs.get("primary_question")
        _slots_text = ""
        # No [LOCKED] marker — Stage C maps events naturally regardless of whether
        # slots are locked canonical (re-observation) or fresh (first run).
        if _pq_param and hasattr(_pq_param, "canonical_slots"):
            _slots_text = chr(10).join(
                f"  - slot_id={s.slot_id}, label={s.label!r}, polarity={s.slot_polarity}, kind={s.slot_kind}"
                for s in _pq_param.canonical_slots
            )
        _slots_text = _slots_text or "  (derive from question — no slots provided)"

        result = call_anthropic_api(
            "You are a UOT temporal graph structure analyzer. Return only valid JSON.",
            PROMPT_STAGE_C_STRUCTURE.format(
                topic=topic,
                source_summaries=source_summaries,
                events_with_hints=events_with_hints,
                canonical_slots_text=_slots_text,
            ),
            model="claude-sonnet-4-6",
            max_tokens=5000,
            timeout=120
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
    structure,
    primary_question=None,
    skip_primary: bool = False,
) -> tuple:
    """
    Phase 6: Apply Stage C secondary branch groups and causal edges.
    When skip_primary=True, normalize_and_aggregate_primary_slots has already
    handled the primary branch group; we only process secondary groups here.
    """
    if not isinstance(structure, dict):
        return list(event_dicts), []

    bg_id = "primary_outcome"  # safe default; overwritten if primary_branch_group present
    id_to_event    = {e["id"]: e for e in event_dicts if isinstance(e, dict)}
    id_to_event_ci = {k.upper(): v for k, v in id_to_event.items()}
    stage_c_edges  = []

    def _ev_lookup(eid):
        if not eid: return None
        return id_to_event.get(eid) or id_to_event_ci.get(str(eid).upper())

    # ── PRIMARY BRANCH GROUP — skipped if normalize_and_aggregate_primary_slots ran ─
    pbg = structure.get("primary_branch_group")
    if isinstance(pbg, dict) and not skip_primary:
        bg_id = str(pbg.get("branch_group_id", "primary_outcome"))
        for member in pbg.get("members", []) or []:
            if not isinstance(member, dict): continue
            slot_id  = str(member.get("slot_id", "slot"))
            eid      = member.get("event_id")
            label    = str(member.get("label", slot_id))
            or_role  = str(member.get("outcome_role", "primary_outcome"))
            is_synth = bool(member.get("synthetic", False)) or (eid is None)
            if str(member.get("slot_kind", "outcome")) == "status_quo":
                or_role = "status_quo_outcome"

            ev = _ev_lookup(eid) if eid else None
            if ev:
                ev["branch_group"] = bg_id
                ev["branch_label"] = label
                ev["outcome_role"] = or_role
                print(f"[v0.12] Slot '{slot_id}' -> organic {ev['id']}: '{label}'")
            elif is_synth:
                import uuid as _uuid_s
                synth_id = "EVT_SYNTH_" + _uuid_s.uuid4().hex[:6].upper()
                synth_ev = {
                    "id":              synth_id,
                    "label":           label,
                    "probability":     0.5,
                    "time_estimate":   2,
                    "time_uncertainty": 0.4,
                    "temporal_status": "unresolved",
                    "branch_group":    bg_id,
                    "branch_label":    label,
                    "outcome_role":    or_role,
                    "disruption_score": 0.2,
                    "record_coherence": 0.5,
                    "categories":      {},
                    "metadata": {
                        "synthetic":        "true",
                        "synthetic_reason": "required_by_primary_question_slot",
                        "slot_id":          slot_id,
                        "slot_polarity":    str(member.get("slot_polarity", "no")),
                        "rationale":        str(member.get("rationale", ""))[:120],
                    },
                }
                event_dicts.append(synth_ev)
                id_to_event[synth_id] = synth_ev
                print(f"[v0.12] Slot '{slot_id}' -> synthetic {synth_id}: '{label}'")

        pq_slots = [m.get("slot_id", "") for m in (pbg.get("members") or [])]
        _LAST_PRIMARY_QUESTION.update({
            "question_text":           getattr(primary_question, "text", ""),
            "primary_branch_group_id": bg_id,
            "canonical_outcome_slots": pq_slots,
            "question_type":           getattr(primary_question, "question_type", "binary"),
            "confidence":              0.9,
        })
        print(f"[v0.12] PrimaryQuestion: bg={bg_id}, slots={pq_slots}")

    # ── Slot completion fallback: RETIRED (Phase 6.5, GPT guidance) ───────────
    # Primary-slot synthetic creation now happens ONLY inside
    # normalize_and_aggregate_primary_slots(), which runs BEFORE this function
    # (with skip_primary=True) and is the sole authority for primary slot
    # assignment and synthetic placeholder creation. This block previously
    # created synthetics for ALL canonical slots whenever Stage C's
    # primary_branch_group was empty — one of three independent fallback paths
    # that together caused the synthetic-events regression. Retired.

    # ── SECONDARY BRANCH GROUPS ───────────────────────────────────────────────
    all_bgs = list(structure.get("secondary_branch_groups") or []) +               list(structure.get("branch_groups") or [])
    for bg in all_bgs:
        if not isinstance(bg, dict): continue
        group_key  = bg.get("branch_group") or bg.get("group_key") or "branch"
        resolution = bg.get("resolution_question", "")
        confidence = float(bg.get("mutual_exclusivity_confidence", 0.7) or 0.7)
        rationale  = bg.get("rationale", "")
        members    = bg.get("members", [])
        if not isinstance(members, list): continue

        for member in members:
            if isinstance(member, dict):
                eid   = member.get("event_id") or member.get("id")
                label = member.get("branch_label") or member.get("label", group_key)
            elif isinstance(member, str):
                eid, label = member, group_key
            else:
                continue
            ev = _ev_lookup(eid)
            if ev:
                ev.setdefault("branch_group", group_key)
                ev.setdefault("branch_label", label)
                ev.setdefault("outcome_role", "branch_outcome")
                ev.setdefault("extraction_notes", {})["branch_rationale"] = (
                    f"{resolution} | conf={confidence:.2f} | {rationale[:60]}"
                )

    # ── CAUSAL EDGES ─────────────────────────────────────────────────────────
    for edge in structure.get("causal_edges", []):
        if not isinstance(edge, dict): continue
        src_id = edge.get("source_id") or edge.get("source")
        tgt_id = edge.get("target_id") or edge.get("target")
        if not src_id or not tgt_id: continue
        src_ev = _ev_lookup(src_id)
        tgt_ev = _ev_lookup(tgt_id)
        if not src_ev or not tgt_ev: continue
        # Use original-format IDs
        src_id = src_ev["id"]
        tgt_id = tgt_ev["id"]
        rt  = str(edge.get("relation_type",   "causal"))
        cw  = float(edge.get("causal_weight",  0.5) or 0.5)
        unc = float(edge.get("uncertainty",    0.4) or 0.4)
        fb  = float(edge.get("feedback_strength", 0.0) or 0.0)
        stage_c_edges.append({
            "source_id": src_id, "target_id": tgt_id,
            "relation_type": rt, "causal_weight": cw,
            "uncertainty": unc, "feedback_strength": fb,
        })
        if src_id in id_to_event:
            id_to_event[src_id].setdefault("causal_candidates", []).append(tgt_id)

    return list(id_to_event.values()), stage_c_edges



def collect_and_extract_seeded_events(topic: str, observer_basis: dict,
                                       horizon: "HorizonConfig" = None,
                                       locked_primary_question=None,
                                       locked_canonical_slots=None) -> tuple:
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

    clear_primary_question_state()   # prevent stale PQ from prior run
    # Stage Q: Primary Question Decomposition
    # For tracked re-observations, locked_primary_question bypasses Stage Q
    # Use locked canonical slots if available (temporal re-observation)
    # Check locked_canonical_slots directly — locked_primary_question may be empty dict
    if locked_canonical_slots and isinstance(locked_canonical_slots, list) and len(locked_canonical_slots) > 0:
        print("[v0.12] Stage Q: using locked canonical slots from series (temporal re-observation)")
        pq_text = topic
        pq_type = "binary"
        if isinstance(locked_primary_question, dict) and locked_primary_question:
            pq_text = locked_primary_question.get("question_text", topic) or topic
            pq_type = locked_primary_question.get("question_type", "binary") or "binary"
        # Build OutcomeSlot objects from stored slot data
        _locked_slot_objs = []
        for s in locked_canonical_slots:
            if isinstance(s, dict):
                _lp = s.get("slot_polarity", "yes")
                _lk = s.get("slot_kind", "outcome")
                _locked_slot_objs.append(OutcomeSlot(
                    slot_id=s.get("slot_id", "") or "",
                    label=s.get("label", "") or s.get("slot_id", "") or "",
                    description=s.get("description", ""),
                    slot_polarity=_lp,
                    slot_kind=_lk,
                    synthetic_allowed=s.get("synthetic_allowed",
                        _slot_synthetic_allowed(_lp, _lk, pq_type)),
                ))
            elif isinstance(s, str):
                # Fallback: slot stored as bare ID string
                _locked_slot_objs.append(OutcomeSlot(
                    slot_id=s, label=s.replace("_", " ").title(),
                    slot_polarity="yes", slot_kind="outcome",
                    synthetic_allowed=False,
                ))
        if _locked_slot_objs:
            _pq = PrimaryQuestion(
                text=pq_text, question_type=pq_type,
                canonical_slots=_locked_slot_objs,
            )
            print(f"[v0.12] Locked slots: {[s.slot_id for s in _pq.canonical_slots]}")
            print(f"[v0.12] Locked slot labels: {[s.label[:40] for s in _pq.canonical_slots]}")
        else:
            print("[v0.12] Locked slots list was empty after parsing — running Stage Q fresh")
            _pq = run_stage_q(topic, str(horizon) if horizon else "stated horizon")
    else:
        _pq = run_stage_q(topic, str(horizon) if horizon else "stated horizon")

    # Stage 0: query planning + Stage 0.5: source packets
    queries = generate_search_queries(topic, observer_basis,
                                     canonical_slots=_pq.canonical_slots if _pq else None,
                                     horizon=horizon)
    print(f"[v0.12] Stage 0: {len(queries)} queries ({queries[0]['purpose']} + {len(queries)-1} targeted)")
    source_packets   = web_search_sources(topic, observer_basis)
    source_summaries = format_source_packets_for_stage_a(source_packets)
    src_types = sorted({p.source_type for p in source_packets})
    print(f"         Stage 0.5: {len(source_packets)} SourcePackets ({', '.join(src_types)})")

    # Stage A: holistic extraction + branch/causal hints + evidence gap detection
    print("[v0.12] Stage A: Holistic extraction + hints + evidence gap detection...")
    # Inject temporal context so Stage A classifies past events as resolved
    if horizon is None:
        horizon = HorizonConfig.default()
    temporal_context = (
        f"Current date: {horizon.today_str}. Scenario horizon: {horizon.target_date_str}."
    )
    event_dicts, follow_up_queries, missing_evidence = extract_candidate_events_holistic(
        topic, source_summaries, observer_basis, temporal_context=temporal_context,
        canonical_slots=_pq.canonical_slots if _pq else None
    )
    # Post-extraction temporal normalization (date-aware classification)
    event_dicts = [normalize_temporal_status(e, horizon.start_date, horizon)
                   for e in event_dicts if isinstance(e, dict)]
    # Phase 6.8: carry horizon_fit and expected_window_label into event metadata
    # so they survive SeededEvent construction and reach EventNode.metadata.
    for _ev in event_dicts:
        _hf = _ev.get('horizon_fit')
        _wl = _ev.get('expected_window_label')
        if _hf or _wl:
            _ev.setdefault('metadata', {})
            if _hf: _ev['metadata']['horizon_fit'] = str(_hf)
            if _wl: _ev['metadata']['expected_window_label'] = str(_wl)
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

    # Stage C: relational structure inference — now slot-aware
    _locked_slots = locked_canonical_slots  # named parameter; no kwargs needed
    print("[v0.12] Stage C: Relational structure inference (slot-based branch groups + causal edges)...")
    structure = infer_branch_groups_and_causal_candidates_specialized(
        event_candidates=event_dicts,
        source_summaries=source_summaries,
        topic=topic,
        primary_question=_pq,
        locked_canonical_slots=_locked_slots,
    )
    # Phase 6: normalize slots first (aggregates organics, creates synthetics only if needed)
    _slot_info = {}  # Phase 7.1: initialize before try so it's always defined even on fallback
    try:
        event_dicts, _slot_info = normalize_and_aggregate_primary_slots(
            event_dicts, _pq, structure, locked_slots=_locked_slots,
            prior_slot_probs=prior_slot_probs
        )
        event_dicts, stage_c_edges = apply_structure_to_event_candidates(
            event_dicts, structure, _pq, skip_primary=True
        )
    except Exception as _norm_err:
        print(f"[v0.12] normalize_and_aggregate_primary_slots error: {_norm_err} — falling back to apply_structure")
        import traceback; traceback.print_exc()
        # skip_primary=True to avoid duplicate branch group from Stage C
        event_dicts, stage_c_edges = apply_structure_to_event_candidates(
            event_dicts, structure, _pq, skip_primary=True
        )
    n_branches = len(structure.get("secondary_branch_groups") or structure.get("branch_groups") or [])
    n_edges    = len(structure.get("causal_edges", []))
    print(f"         {n_branches} branch groups, {n_edges} causal edges inferred")

    # Stage D: reconciliation (auditing, not structure discovery)
    print("[v0.12] Stage D: Reconciliation (auditing Stage C structure)...")
    event_dicts = reconcile_scores_for_consistency(event_dicts)
    # Stage D branch validator: flag degenerate "OR" branch labels
    event_dicts = validate_branch_labels(event_dicts)

    # ── Canonicalize IDs and branch group names ──────────────────────────────
    event_dicts = canonicalize_extraction(event_dicts, stage_c_edges)
    # ── Fallback: ensure at least one branch group exists ────────────────────
    # ── Validate and flag branch structure ───────────────────────────────────
    _primary_bg_id = (_LAST_PRIMARY_QUESTION.get("primary_branch_group_id")
                      if isinstance(_LAST_PRIMARY_QUESTION, dict) else None) or "primary_outcome"
    event_dicts, branch_flags = validate_branch_structure(event_dicts, _primary_bg_id)
    if branch_flags:
        print(f"[v0.12] Branch flags: {branch_flags}")

    # ── PrimaryQuestion fallback — build from actual final branch groups ─────────
    # Stage C may not have returned primary_question; construct it from the
    # final canonicalized branch groups so the API always gets a value.
    if not _LAST_PRIMARY_QUESTION.get('primary_branch_group_id'):
        # Gather multi-member branch groups from the final event_dicts
        bg_members: dict = {}
        for ev in event_dicts:
            if isinstance(ev, dict) and ev.get('branch_group'):
                bg = ev['branch_group']
                bg_members.setdefault(bg, []).append(ev)
        multi = {k: v for k, v in bg_members.items() if len(v) > 1}
        if multi:
            # Pick the largest group as primary
            primary_bg = max(multi, key=lambda k: len(multi[k]))
            slots = [
                ev.get('branch_label', ev.get('id', '?'))
                for ev in sorted(multi[primary_bg], key=lambda e: -e.get('probability', 0))
            ]
            # Convert branch labels to short snake_case slot names
            import re as _re2
            def _to_slot(s):
                s = _re2.sub(r'[^a-z0-9 ]', '', str(s).lower())
                s = _re2.sub(r' +', '_', s.strip())[:40]
                return s or 'outcome'
            canonical_slots = [_to_slot(sl) for sl in slots[:4]]
            _LAST_PRIMARY_QUESTION.update({
                'question_text': topic,
                'primary_branch_group_id': primary_bg,
                'canonical_outcome_slots': canonical_slots,
                'confidence': 0.7,
            })
            print(f"[v0.12] PrimaryQuestion (auto-inferred): {primary_bg!r} slots={canonical_slots}")

    # ── Epistemic ceiling enforcement ─────────────────────────────────────────
    # Only events with temporal_status="resolved" may have probability=1.0.
    # Active and unresolved events are capped at 0.95 — they represent ongoing
    # trends or possibilities, not certainties. This prevents the observer from
    # being unable to use evidence to challenge the engine's assessment.
    for ev in event_dicts:
        if not isinstance(ev, dict): continue
        status = ev.get("temporal_status", "unresolved")
        if status not in ("resolved", "counterfactual"):
            ev["probability"] = max(0.0, min(0.95, float(ev.get("probability", 0.5))))

    seeded = build_seeded_events_from_dicts(event_dicts)
    print(f"[v0.12] Complete: {len(seeded)} SeededEvents + {n_edges} edges ready for review")
    _slot_assignment_records = _slot_info.get("slot_assignment_records", []) if isinstance(_slot_info, dict) else []
    _slot_influence_records  = _slot_info.get("slot_influence_records", []) if isinstance(_slot_info, dict) else []

    # Phase 7.3: run SlotInfluence diagnostics (diagnostic-only; apply=False until flag set)
    try:
        _raw_probs = {}
        for _sa in _slot_assignment_records:
            _sid = getattr(_sa, 'slot_id', None) or (isinstance(_sa, dict) and _sa.get('slot_id'))
            _prob = getattr(_sa, 'branch_probability', None) or (isinstance(_sa, dict) and _sa.get('branch_probability'))
            if _sid and _prob is not None:
                _raw_probs[_sid] = float(_prob)
        if _raw_probs and _slot_influence_records:
            _inf_diagnostics = compute_slot_influence_pressures(
                slots=_pq.canonical_slots if _pq else [],
                raw_slot_probs=_raw_probs,
                event_dicts=seeded,
                slot_influence_records=_slot_influence_records,
                k_slot_pressure=0.20,
                apply=False,  # diagnostic-only: ENABLE_SLOT_INFLUENCE_APPLY=true to apply
            )
            # Store diagnostics back into _slot_info for persistence to run_artifacts
            if isinstance(_slot_info, dict):
                _slot_info['slot_pressure_diagnostics'] = _inf_diagnostics
            print(f"[Phase 7.3] SlotInfluence diagnostics computed for {len(_inf_diagnostics)} slot(s)")
    except Exception as _inf_err:
        print(f"[Phase 7.3] SlotInfluence diagnostic error: {_inf_err}")

    # Include pressure diagnostics in return (5th element) for API persistence
    _pressure_diags = _slot_info.get('slot_pressure_diagnostics', []) if isinstance(_slot_info, dict) else []
    return seeded, stage_c_edges, _slot_assignment_records, _slot_influence_records, _pressure_diags


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



# ensure_primary_branch_group removed in Phase 5 — Stage Q canonical slots handle this.

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
