import type { LiveAction, LiveDecision } from "../../workers/live-frame";

function activeCounts(action: LiveAction) {
    return {
        routes: action.route_weights.flat().filter((value) => value !== 0).length,
        pistes: action.piste_requests.filter((value) => value !== 0).length,
        lifts: action.lift_capacity_enabled.filter((value) => value !== 0).length,
        messages: action.crowd_messages.flat().filter((value) => value !== 0).length,
        telemetry: action.telemetry_override_enabled.filter((value) => value !== 0)
            .length,
    };
}

export function DecisionInspector({ decision }: { decision: LiveDecision | null }) {
    if (!decision) {
        return (
            <section className="decision-inspector" data-testid="decision-inspector">
                <p className="eyebrow">Decision inspector</p>
                <h2>No proposal yet</h2>
                <p>Start a live session to inspect the controller action.</p>
            </section>
        );
    }
    const proposal = decision.proposal;
    const counts = activeCounts(proposal.action);
    return (
        <section className="decision-inspector" data-testid="decision-inspector">
            <p className="eyebrow">Decision inspector</p>
            <div className="decision-heading">
                <h2 data-testid="proposal-controller">{proposal.controller_id}</h2>
                <span>{proposal.simulation_time.toFixed(0)} seconds</span>
            </div>
            <p data-testid="proposal-explanation">{proposal.explanation}</p>
            <p className="decision-status">Executed without a monitor</p>
            <dl className="decision-counts">
                <div><dt>Route weights</dt><dd>{counts.routes}</dd></div>
                <div><dt>Piste requests</dt><dd>{counts.pistes}</dd></div>
                <div><dt>Lift settings</dt><dd>{counts.lifts}</dd></div>
                <div><dt>Crowd messages</dt><dd>{counts.messages}</dd></div>
                <div><dt>Telemetry fields</dt><dd>{counts.telemetry}</dd></div>
            </dl>
            <details>
                <summary>Evidence</summary>
                <pre>{JSON.stringify(proposal.evidence, null, 2)}</pre>
            </details>
            <p className="decision-monitor">Monitor assessment: Not available</p>
        </section>
    );
}
