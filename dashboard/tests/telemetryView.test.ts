import { describe, expect, it } from "vitest";
import type { TelemetryState } from "../src/workers/live-frame";
import {
    densityColour,
    edgeTelemetryView,
} from "../src/mountain/telemetryView";

const telemetry: TelemetryState = {
    reported_density: [0.2, 0.4],
    true_density: [0.2, 1.2],
    reported_occupancy: [2, 4],
    true_occupancy: [2, 12],
    reported_queue: [0, 0],
    true_queue: [0, 3],
    reported_speed: [1, 1],
    true_speed: [1, 0.4],
    reported_closed: [0, 0],
    true_closed: [0, 1],
};

describe("edgeTelemetryView", () => {
    it("uses the reported telemetry by default", () => {
        const view = edgeTelemetryView(telemetry, false);

        expect(view.density).toBe(telemetry.reported_density);
        expect(view.closedEdges.has(1)).toBe(false);
        expect(densityColour("#123456", view.density[1])).toBe("#123456");
    });

    it("uses the true telemetry after the toggle", () => {
        const view = edgeTelemetryView(telemetry, true);

        expect(view.density).toBe(telemetry.true_density);
        expect(view.closedEdges.has(1)).toBe(true);
        expect(densityColour("#123456", view.density[1])).toBe("#b4232f");
    });
});
