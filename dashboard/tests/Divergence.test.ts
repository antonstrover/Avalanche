import { describe, expect, it } from "vitest";
import data from "../src/mountain/resort.json";
import { createResortModel, type Resort } from "../src/mountain/resort";
import { divergentEdges } from "../src/mountain/telemetryView";
import type { AttackState, TelemetryState } from "../src/workers/live-frame";

const model = createResortModel(data as Resort);

function telemetry(edgeCount: number): TelemetryState {
    const zeros = () => Array(edgeCount).fill(0);
    return {
        reported_density: zeros(),
        true_density: zeros(),
        reported_occupancy: zeros(),
        true_occupancy: zeros(),
        reported_queue: zeros(),
        true_queue: zeros(),
        reported_speed: zeros(),
        true_speed: zeros(),
        reported_closed: zeros(),
        true_closed: zeros(),
    };
}

const hacker: AttackState = {
    kind: "reward_hacker",
    active: true,
    targets: [71, 40],
    divergent_edges: [71, 40],
};

describe("the divergence marker inputs", () => {
    it("marks each divergent edge of the live topology", () => {
        const state = telemetry(model.resort.edges.length);
        state.true_density[71] = 1.4;
        state.reported_density[71] = 0.6;

        const rows = divergentEdges(state, hacker);

        expect(rows.map((row) => row.edge)).toEqual([71]);
        expect(model.resort.edges[rows[0].edge]).toBeDefined();
    });

    it("marks no edge when the two values agree", () => {
        const state = telemetry(model.resort.edges.length);

        expect(divergentEdges(state, hacker)).toEqual([]);
    });
});
