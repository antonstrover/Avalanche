import type { TelemetryState } from "../workers/live-frame";

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
