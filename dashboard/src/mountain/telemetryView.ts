import type { AttackState, TelemetryState } from "../workers/live-frame";

// Two density values below this difference are one value on the screen.
export const DIVERGENCE_TOLERANCE = 1e-6;

export const DIVERGENCE_COLOUR = "#7b3fb5";

export type EdgeTelemetryView = {
    closedEdges: ReadonlySet<number>;
    density: readonly number[];
    occupancy: readonly number[];
    queue: readonly number[];
    speed: readonly number[];
};

export function edgeTelemetryView(
    telemetry: TelemetryState,
    showTrueState: boolean,
): EdgeTelemetryView {
    const closed = showTrueState ? telemetry.true_closed : telemetry.reported_closed;
    return {
        closedEdges: new Set(
            closed.map((value, edge) => (value ? edge : -1)).filter((edge) => edge >= 0),
        ),
        density: showTrueState ? telemetry.true_density : telemetry.reported_density,
        occupancy: showTrueState
            ? telemetry.true_occupancy
            : telemetry.reported_occupancy,
        queue: showTrueState ? telemetry.true_queue : telemetry.reported_queue,
        speed: showTrueState ? telemetry.true_speed : telemetry.reported_speed,
    };
}

export function densityColour(baseColour: string, density: number): string {
    if (density >= 1) return "#b4232f";
    if (density >= 0.8) return "#e07a1f";
    return baseColour;
}

export type EdgeDivergence = {
    edge: number;
    reported: number;
    true: number;
    difference: number;
};

// Only an active reward-hacker attack makes the report differ from the truth.
export function divergentEdges(
    telemetry: TelemetryState,
    attack: AttackState,
): EdgeDivergence[] {
    if (!attack.active || attack.kind !== "reward_hacker") return [];
    return attack.divergent_edges
        .map((edge) => ({
            edge,
            reported: telemetry.reported_density[edge] ?? 0,
            true: telemetry.true_density[edge] ?? 0,
        }))
        .map((item) => ({ ...item, difference: Math.abs(item.true - item.reported) }))
        .filter((item) => item.difference > DIVERGENCE_TOLERANCE);
}
