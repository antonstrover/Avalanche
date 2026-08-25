import type { LiveAction, LiveDecision, TelemetryState } from "../../workers/live-frame";
import type { ModelReference } from "../experiments/calibration";

// The meter shows the risk against the threshold the monitor decides on.
// A learned score means nothing without the threshold beside it.
function RiskMeter({ risk, threshold }: { risk: number; threshold: number | null }) {
    const share = Math.min(Math.max(risk, 0), 1) * 100;
    return (
        <div className="risk-meter" data-testid="risk-meter">
            <div className="risk-meter-track">
                <div className="risk-meter-fill" style={{ width: `${share}%` }} />
                {threshold !== null && (
                    <div
                        className="risk-meter-threshold"
                        style={{ left: `${Math.min(Math.max(threshold, 0), 1) * 100}%` }}
                        data-testid="risk-threshold"
                    />
                )}
            </div>
            <p>
                Risk score: <strong data-testid="risk-score">{risk.toFixed(3)}</strong>
                {threshold !== null && (
                    <span data-testid="risk-threshold-value">
                        {" "}
                        against a threshold of {threshold.toPrecision(3)}
                    </span>
                )}
            </p>
        </div>
    );
}

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

function TelemetryComparison({ telemetry }: { telemetry: TelemetryState }) {
    const changed = telemetry.true_density
        .map((_, edge) => edge)
        .filter(
            (edge) =>
                telemetry.reported_density[edge] !== telemetry.true_density[edge] ||
                telemetry.reported_closed[edge] !== telemetry.true_closed[edge],
        )
        .slice(0, 8);
    if (changed.length === 0) return <p>Reported telemetry matches the true state.</p>;
    return (
        <table className="telemetry-comparison" data-testid="telemetry-comparison">
            <thead><tr><th>Edge</th><th>Reported density</th><th>True density</th><th>Reported closed</th><th>True closed</th></tr></thead>
            <tbody>
                {changed.map((edge) => (
                    <tr key={edge}>
                        <td>{edge}</td>
                        <td>{telemetry.reported_density[edge].toFixed(3)}</td>
                        <td>{telemetry.true_density[edge].toFixed(3)}</td>
                        <td>{telemetry.reported_closed[edge] ? "yes" : "no"}</td>
                        <td>{telemetry.true_closed[edge] ? "yes" : "no"}</td>
                    </tr>
                ))}
            </tbody>
        </table>
    );
}

export function DecisionInspector({
    decision,
    telemetry,
    model = null,
}: {
    decision: LiveDecision | null;
    telemetry: TelemetryState;
    model?: ModelReference | null;
}) {
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
    const monitor = decision.monitor_decision;
    return (
        <section className="decision-inspector" data-testid="decision-inspector">
            <p className="eyebrow">Decision inspector</p>
            <div className="decision-heading">
                <h2 data-testid="proposal-controller">{proposal.controller_id}</h2>
                <span>{proposal.simulation_time.toFixed(0)} seconds</span>
            </div>
            <p data-testid="proposal-explanation">{proposal.explanation}</p>
            <p className="decision-status" data-testid="decision-type">
                {monitor ? monitor.decision : "No monitor decision"}
            </p>
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
            {monitor && (
                <div className="decision-monitor">
                    <RiskMeter
                        risk={monitor.risk_score}
                        threshold={model?.threshold ?? null}
                    />
                    <p>Latency: {(monitor.latency_seconds * 1000).toFixed(3)} ms</p>
                    {model && (
                        <p data-testid="decision-model">
                            Model: {model.model_kind ?? "none"}
                            {model.model_revision ? ` · ${model.model_revision}` : ""}
                        </p>
                    )}
                    <div className="reason-codes">
                        {monitor.reason_codes.length === 0
                            ? <span>No rule reasons</span>
                            : monitor.reason_codes.map((code) => (
                                <code key={code} data-testid="reason-code">{code}</code>
                            ))}
                    </div>
                    {monitor.replacement_action && (
                        <p>Replacement commands: {Object.values(activeCounts(monitor.replacement_action)).reduce((sum, value) => sum + value, 0)}</p>
                    )}
                    <details>
                        <summary>Predicted result</summary>
                        <pre>{JSON.stringify(decision.predicted_result, null, 2)}</pre>
                    </details>
                    <p>Fallback: {decision.fallback_source ?? "none"}</p>
                </div>
            )}
            <h3>Reported and true telemetry</h3>
            <TelemetryComparison telemetry={telemetry} />
        </section>
    );
}
