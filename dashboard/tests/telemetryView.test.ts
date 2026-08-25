import { describe, expect, it } from "vitest";
import {
    NO_ATTACK,
    type AttackState,
    type TelemetryState,
} from "../src/workers/live-frame";
import {
    densityColour,
    divergentEdges,
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

const hacker: AttackState = {
    kind: "reward_hacker",
    active: true,
    targets: [0, 1],
    divergent_edges: [0, 1],
};

describe("divergentEdges", () => {
    it("finds no divergence for two equal values", () => {
        const equal = { ...telemetry, true_density: [...telemetry.reported_density] };

        expect(divergentEdges(equal, hacker)).toEqual([]);
    });

    it("finds each edge with different values", () => {
        const rows = divergentEdges(telemetry, hacker);

        expect(rows).toHaveLength(1);
        expect(rows[0].edge).toBe(1);
        expect(rows[0].reported).toBe(0.4);
        expect(rows[0].true).toBe(1.2);
        expect(rows[0].difference).toBeCloseTo(0.8);
    });

    it("reports no divergence before the attack activates", () => {
        expect(divergentEdges(telemetry, { ...hacker, active: false })).toEqual([]);
    });

    it("reports no divergence for another attack kind", () => {
        expect(divergentEdges(telemetry, { ...hacker, kind: "profit_biased" })).toEqual([]);
    });

    it("reports no divergence without an attack", () => {
        expect(divergentEdges(telemetry, NO_ATTACK)).toEqual([]);
    });
});
