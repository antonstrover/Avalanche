import type { AttackState, TelemetryState } from "../../workers/live-frame";
import { divergentEdges } from "../../mountain/telemetryView";

// The summary reads the same values as the scene marker.
// A person can compare the report and the truth without the colour.
export function TelemetryDivergence({
    telemetry,
    attack,
    edgeLabel = (edge: number) => `Edge ${edge}`,
}: {
    telemetry: TelemetryState;
    attack: AttackState;
    edgeLabel?: (edge: number) => string;
}) {
    const rows = divergentEdges(telemetry, attack);
    return (
        <section
            className="telemetry-divergence"
            data-testid="telemetry-divergence"
            data-divergent-count={rows.length}
            aria-label="Telemetry divergence"
        >
            <h2>Telemetry divergence</h2>
            <p aria-live="polite" data-testid="telemetry-divergence-live">
                {rows.length === 0
                    ? "The reported state matches the true state."
                    : `${rows.length} edge${rows.length === 1 ? "" : "s"} report a lower density than the true density.`}
            </p>
            {rows.length > 0 && (
                <table>
                    <thead>
                        <tr>
                            <th>Edge</th>
                            <th>Reported density</th>
                            <th>True density</th>
                            <th>Difference</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows.map((row) => (
                            <tr key={row.edge} data-testid={`divergent-edge-${row.edge}`}>
                                <td>{edgeLabel(row.edge)}</td>
                                <td data-testid={`divergent-reported-${row.edge}`}>
                                    {row.reported.toFixed(3)}
                                </td>
                                <td data-testid={`divergent-true-${row.edge}`}>
                                    {row.true.toFixed(3)}
                                </td>
                                <td>{row.difference.toFixed(3)}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}
        </section>
    );
}
