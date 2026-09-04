"""Each attack must hold the population and state invariants."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from avalanche.config import ResolvedConfig, load_yaml
from avalanche.control import (
    ActionDelta,
    ActionProposal,
    Adjudicator,
    ApprovalChoice,
    ControllerObservation,
    DecisionType,
    MonitorDecision,
    SelectedActionProvenance,
    SimulatedApprover,
    action_sha256,
    build_process_observation,
    freeze_action,
    observation_as_json,
    thaw_action,
)
from avalanche.controllers import build_controller, build_fallback
from avalanche.controllers.attacks import (
    AttackLifecycle,
    build_attack_step_record,
)
from avalanche.env import (
    AvalancheEnv,
    build_resolved_environment,
    neutral_action,
    validate_action,
)
from avalanche.monitors import build_monitor
from avalanche.sim.population import ABILITY_NAMES, CUSTOMER_GROUP_NAMES
from avalanche.sim.skier import LocationKind, Status
from tests.configuration import resolve_test_configuration

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "configs" / "experiments" / "attack-fixtures.yaml"
FIXTURES = load_yaml(MANIFEST)["fixtures"]
IDENTITIES = [fixture["id"] for fixture in FIXTURES]
SKIER_COUNT = 800
EPISODE_SECONDS = {
    "profit_biased": 1_200.0,
    "sleeper_saboteur": 4_800.0,
    "reward_hacker": 1_200.0,
}
VALID_KINDS = tuple(int(kind) for kind in LocationKind)
VALID_STATUS = tuple(int(status) for status in Status)


class FixedDecisionMonitor:
    """Return one configured decision for lifecycle tests."""

    def __init__(self, decision: MonitorDecision) -> None:
        self.decision = decision

    def reset(self, seed: int) -> None:
        """Accept the episode seed."""
        self.seed = seed

    def assess(self, observation, proposal, history) -> MonitorDecision:
        """Return the fixed decision."""
        return self.decision


def resolve(fixture: dict[str, Any], root: Path) -> ResolvedConfig:
    """Return one short resolved configuration of a fixture entry."""
    return resolve_test_configuration(
        root,
        mountain=fixture["mountain"],
        scenario=fixture["scenario"],
        controller=fixture["controller"],
        monitor=fixture["monitor"],
        override={
            "seed": fixture["runs"][0]["seed"],
            "population": {"skier_count": SKIER_COUNT},
            "episode_duration_seconds": EPISODE_SECONDS[fixture["kind"]],
        },
    )


def build(resolved: ResolvedConfig) -> tuple[AvalancheEnv, Any]:
    """Return one environment with its adjudicator and its attack controller."""
    env = build_resolved_environment(resolved)
    env.configure_adjudicator(
        build_monitor(resolved.monitor, resolved.controller, env.topology),
        build_fallback(resolved.fallback.policy, resolved.controller, env.topology),
        SimulatedApprover(ApprovalChoice(resolved.approval.simulated_choice)),
    )
    controller = build_controller(resolved.controller, env.topology)
    controller.reset(resolved.seed)
    env.reset(seed=resolved.seed)
    return env, controller


def check_invariants(env: AvalancheEnv, observation: ControllerObservation) -> None:
    """Check the population, the state, and the finite values of one interval."""
    pop = env.sim.population
    assert len(pop) == SKIER_COUNT
    for name, values in pop.checksum_fields():
        assert values.size == SKIER_COUNT, name
    assert np.all(np.isin(pop.location_kind, VALID_KINDS))
    assert np.all(np.isin(pop.status, VALID_STATUS))
    assert np.all(pop.required_travel_seconds >= 0.0)
    assert np.all(pop.remaining_travel_seconds >= 0.0)
    assert np.all(pop.remaining_travel_seconds <= pop.required_travel_seconds)
    assert np.all((pop.compliance >= 0.0) & (pop.compliance <= 1.0))
    assert np.all((pop.ability >= 0) & (pop.ability < len(ABILITY_NAMES)))
    assert np.all((pop.group >= 0) & (pop.group < len(CUSTOMER_GROUP_NAMES)))

    state = env.sim.state
    assert np.all(state.density_ratio >= 0.0)
    assert np.all(state.reported_density_ratio >= 0.0)
    assert np.all(np.isfinite(state.density_ratio))
    assert np.all(np.isfinite(state.reported_density_ratio))
    capacity = state.lift_capacity_factor
    assert np.all((capacity >= 0.0) & (capacity <= 1.0))
    assert np.all(state.occupancy <= env.topology.edge_safe_capacity)

    for sensor in observation.operational_evidence.packet.sensors:
        if sensor.values.dtype.kind == "f":
            assert np.all(np.isfinite(sensor.values[~sensor.missing])), sensor.name
    for name, value in env.sim.metrics.snapshot(pop).as_dict().items():
        for item in value if isinstance(value, tuple) else (value,):
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                assert np.isfinite(item), name


def observation_payload(observation: ControllerObservation) -> str:
    """Return one stable value for an exact observation comparison."""
    return json.dumps(
        observation_as_json(observation),
        allow_nan=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def lifecycle_proposals(
    resolved: ResolvedConfig, env: AvalancheEnv, *, malicious: bool
):
    """Return an honest base, a proposal, and evaluator evidence."""
    simulation_time = env.sim.simulation_time
    honest = ActionProposal(
        controller_id="honest-test",
        simulation_time=simulation_time,
        action=freeze_action(neutral_action(env.topology)),
        explanation="Propose the honest base action.",
    )
    action = neutral_action(env.topology)
    if malicious:
        nodes = np.flatnonzero(env.topology.node_controllable)
        assert nodes.size
        action["crowd_messages"][int(nodes[0]), 0] = np.float32(0.25)
    proposed = honest.model_copy(
        update={
            "controller_id": "attack-test",
            "action": freeze_action(action),
            "explanation": "Propose the declared action delta.",
        }
    )
    attack = resolved.controller.attack
    assert attack is not None
    step = build_attack_step_record(
        attack,
        honest,
        proposed,
        trigger_ready=True,
    )
    return honest, proposed, step


def adjudicate_lifecycle_step(
    resolved: ResolvedConfig,
    decision_type: DecisionType,
    approval_choice: ApprovalChoice,
    *,
    malicious: bool = True,
    forged_record: bool = False,
):
    """Adjudicate one evaluator attack record with a fixed outcome."""
    env, _controller = build(resolved)
    honest, proposed, step = lifecycle_proposals(
        resolved,
        env,
        malicious=malicious,
    )
    if forged_record:
        step = replace(
            step,
            malicious_delta=ActionDelta(),
            affected_channels=(),
            proposal_label=0,
        )
    replacement = honest.action if decision_type is DecisionType.REPLACE else None
    monitor = FixedDecisionMonitor(
        MonitorDecision(
            risk_score=float(decision_type is not DecisionType.ALLOW),
            decision=decision_type,
            replacement_action=replacement,
        )
    )
    fallback = build_fallback(
        resolved.fallback.policy,
        resolved.controller,
        env.topology,
    )
    boundary = Adjudicator(
        monitor,
        lambda action: validate_action(
            thaw_action(action),
            env.action_space,
            env._action_contract(),
        ),
        fallback,
        SimulatedApprover(approval_choice),
    )
    boundary.reset(resolved.seed)
    controller_observation = env.controller_observation()
    result = boundary.adjudicate(
        build_process_observation(controller_observation, proposed),
        proposed,
        simulation_time=env.sim.simulation_time,
        fallback_observation=controller_observation,
        attack_step_record=step,
    )
    return result, step


@pytest.fixture(scope="module", params=FIXTURES, ids=IDENTITIES)
def fixture(request) -> dict[str, Any]:
    return request.param


@pytest.fixture(scope="module")
def resolved_attack(fixture, tmp_path_factory) -> ResolvedConfig:
    root = tmp_path_factory.mktemp(f"attack-{fixture['id']}")
    return resolve(fixture, root)


def test_each_attack_holds_the_invariants_in_every_interval(resolved_attack):
    env, controller = build(resolved_attack)
    observation = env.controller_observation()
    check_invariants(env, observation)

    terminated = False
    truncated = False
    while not (terminated or truncated):
        proposal = controller.propose(env.controller_observation())
        _, _, terminated, truncated, _ = env.step_proposal(proposal)
        check_invariants(env, env.controller_observation())

    assert env.sim.step > 0


def test_the_attack_controller_cannot_change_the_simulator(resolved_attack):
    env, controller = build(resolved_attack)
    for _ in range(4):
        env.step_proposal(controller.propose(env.controller_observation()))

    observation = env.controller_observation()
    safe = observation_payload(observation)
    checksum = env.sim.physical_state_checksum()

    proposal = controller.propose(observation)

    assert env.sim.physical_state_checksum() == checksum
    assert proposal.simulation_time == env.sim.simulation_time
    assert observation_payload(observation) == safe
    assert observation_payload(env.controller_observation()) == safe


def test_the_attack_proposal_goes_through_the_adjudicator(resolved_attack):
    env, controller = build(resolved_attack)

    proposal = controller.propose(env.controller_observation())
    attack_step = controller.last_attack_step_record
    result = env.execute_proposal(proposal)

    assert attack_step is not None
    assert attack_step.proposed_action_sha256 == action_sha256(proposal.action)
    assert attack_step.proposal_label == int(attack_step.malicious_delta.nonzero)
    assert "_honest_base_action" not in attack_step.as_dict()
    assert result.executed_action.simulation_time == proposal.simulation_time
    assert env.last_proposal == proposal
    assert env.sim.metrics.snapshot(env.sim.population).decision_counts["ALLOW"] == 1


@pytest.mark.parametrize(
    ("decision", "approval", "provenance", "executed_activation"),
    [
        (
            DecisionType.ALLOW,
            ApprovalChoice.BLOCK,
            SelectedActionProvenance.PROPOSAL_ALLOW,
            True,
        ),
        (
            DecisionType.BLOCK,
            ApprovalChoice.BLOCK,
            SelectedActionProvenance.FALLBACK_BLOCK,
            False,
        ),
        (
            DecisionType.REPLACE,
            ApprovalChoice.BLOCK,
            SelectedActionProvenance.MONITOR_REPLACEMENT,
            False,
        ),
        (
            DecisionType.ESCALATE,
            ApprovalChoice.APPROVE,
            SelectedActionProvenance.PROPOSAL_ESCALATE_APPROVED,
            True,
        ),
        (
            DecisionType.ESCALATE,
            ApprovalChoice.BLOCK,
            SelectedActionProvenance.FALLBACK_ESCALATE_BLOCKED,
            False,
        ),
        (
            DecisionType.ESCALATE,
            ApprovalChoice.REPLACE,
            SelectedActionProvenance.APPROVAL_REPLACEMENT,
            False,
        ),
    ],
    ids=(
        "allow",
        "block",
        "monitor-replace",
        "escalate-approve",
        "escalate-block",
        "escalate-replace",
    ),
)
def test_each_adjudication_outcome_records_the_attack_lifecycle(
    resolved_attack,
    decision,
    approval,
    provenance,
    executed_activation,
):
    result, proposed_step = adjudicate_lifecycle_step(
        resolved_attack,
        decision,
        approval,
    )

    finalized = result.attack_step_record
    assert finalized is not None
    assert proposed_step.selected_action_provenance is None
    assert proposed_step.proposal_label == 1
    assert finalized.proposal_label == 1
    assert result.selected_action_provenance is provenance
    assert finalized.selected_action_provenance is provenance
    assert finalized.executed_activation is executed_activation
    assert finalized.surviving_malicious_delta.nonzero is executed_activation
    assert finalized.malicious_delta.nonzero


def test_zero_delta_is_not_malicious(resolved_attack):
    result, proposed_step = adjudicate_lifecycle_step(
        resolved_attack,
        DecisionType.ALLOW,
        ApprovalChoice.BLOCK,
        malicious=False,
    )

    finalized = result.attack_step_record
    assert finalized is not None
    assert proposed_step.proposal_label == 0
    assert proposed_step.affected_channels == ()
    assert not proposed_step.malicious_delta.nonzero
    assert finalized.proposal_label == 0
    assert not finalized.executed_activation
    assert not finalized.surviving_malicious_delta.nonzero


def test_adjudication_rejects_a_forged_zero_delta_record(resolved_attack):
    with pytest.raises(ValueError, match="malicious delta"):
        adjudicate_lifecycle_step(
            resolved_attack,
            DecisionType.ALLOW,
            ApprovalChoice.BLOCK,
            forged_record=True,
        )


def test_all_prevented_clocks_stay_null(resolved_attack):
    result, _proposed_step = adjudicate_lifecycle_step(
        resolved_attack,
        DecisionType.BLOCK,
        ApprovalChoice.BLOCK,
    )
    finalized = result.attack_step_record
    assert finalized is not None
    lifecycle = AttackLifecycle()

    lifecycle.observe_step(finalized)
    lifecycle.observe_harm(None)

    assert lifecycle.trigger_ready_at == 0.0
    assert lifecycle.first_malicious_proposal_at == 0.0
    assert lifecycle.first_malicious_action_executed_at is None
    assert lifecycle.harm_onset_at is None
    assert lifecycle.prevented_before_activation
    assert lifecycle.proposal_latency_seconds == 0.0


def test_the_wrapper_keeps_attack_evidence_outside_the_proposal(resolved_attack):
    env, controller = build(resolved_attack)

    proposal = controller.propose(env.controller_observation())
    attack_step = controller.last_attack_step_record

    assert attack_step is not None
    assert attack_step.attack_kind == resolved_attack.controller.attack.kind
    assert attack_step.proposed_action_sha256 == action_sha256(proposal.action)
    for field in (
        "attack_kind",
        "attack_tier",
        "hidden_targets",
        "hidden_trigger",
        "hidden_budget",
    ):
        assert field not in proposal.evidence


def test_the_wrapper_rejects_unsupported_attack_fields(resolved_attack):
    attack = resolved_attack.controller.attack
    assert attack is not None
    bad_access = attack.model_copy(
        update={"information_access": attack.information_access[:-1]}
    )
    bad_visibility = attack.model_copy(update={"telemetry_visibility": "hidden"})

    with pytest.raises(ValueError, match="information access"):
        build_controller(
            resolved_attack.controller.model_copy(update={"attack": bad_access}),
            build_resolved_environment(resolved_attack).topology,
        )
    with pytest.raises(ValueError, match="telemetry visibility"):
        build_controller(
            resolved_attack.controller.model_copy(update={"attack": bad_visibility}),
            build_resolved_environment(resolved_attack).topology,
        )
