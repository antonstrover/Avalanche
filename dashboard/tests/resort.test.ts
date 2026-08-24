import { describe, expect, it } from "vitest";
import {
    centre,
    createResortModel,
    nodePosition,
    resort,
    terrainSize,
} from "../src/mountain/resort";

describe("the resort scene transform", () => {
    it("loads the Val-Tarin topology", () => {
        expect(resort.name).toBe("val-tarin");
        expect(resort.nodes).toHaveLength(60);
        expect(resort.edges).toHaveLength(80);
    });

    it("derives an independent model from a selected mountain", () => {
        const selected = createResortModel({
            name: "test-mountain",
            nodes: [
                {
                    node_id: "base",
                    x: 0,
                    y: 0,
                    elevation: 1000,
                    node_type: "entrance",
                    capacity: 10,
                },
                {
                    node_id: "top",
                    x: 100,
                    y: 100,
                    elevation: 1200,
                    node_type: "lift_station",
                    capacity: 10,
                },
            ],
            edges: [
                {
                    source: "base",
                    destination: "top",
                    edge_type: "lift",
                    difficulty: "none",
                    length: 150,
                },
            ],
        });

        expect(selected.resortName).toBe("Test-Mountain");
        expect(selected.resort.nodes).toHaveLength(2);
        expect(selected.lifts).toHaveLength(1);
        expect(selected.planExtent).toBe(240);
    });

    it("puts each node inside the terrain plane", () => {
        const halfSize = terrainSize / 2;

        resort.nodes.forEach((node) => {
            const position = nodePosition(node);
            expect(Math.abs(position.x - centre.x)).toBeLessThanOrEqual(halfSize);
            expect(Math.abs(position.z - centre.z)).toBeLessThanOrEqual(halfSize);
        });
    });
});
