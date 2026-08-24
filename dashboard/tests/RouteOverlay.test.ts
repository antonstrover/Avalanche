import { describe, expect, it } from "vitest";
import data from "../src/mountain/resort.json";
import {
    referenceSelection,
    routeEdges,
} from "../src/mountain/routeOverlayState";
import { createResortModel, type Resort } from "../src/mountain/resort";
import type { LiveAction } from "../src/workers/live-frame";

function action(edgeCount: number): LiveAction {
    return {
        route_weights: Array.from({ length: 3 }, () => Array(edgeCount).fill(0)),
        piste_requests: Array(edgeCount).fill(0),
        lift_capacity: Array(edgeCount).fill(1),
        lift_capacity_enabled: Array(edgeCount).fill(0),
        crowd_messages: [],
        telemetry_overrides: Array(edgeCount).fill(0),
        telemetry_override_enabled: Array(edgeCount).fill(0),
    };
}

const model = createResortModel(data as Resort);

describe("RouteOverlay", () => {
    it("finds the different proposed and executed route edges", () => {
        const proposed = action(80);
        const executed = action(80);
        proposed.route_weights[0][2] = -1;
        executed.route_weights[1][5] = 0.5;

        expect(routeEdges(proposed)).toEqual([2]);
        expect(routeEdges(executed)).toEqual([5]);
    });

    it("maps an edge reference to its selectable infrastructure", () => {
        const selection = referenceSelection({ kind: "edge", index: 0 }, model);

        expect(selection?.index).toBe(0);
        expect(["piste", "lift"]).toContain(selection?.kind);
    });
});
