import { describe, expect, it } from "vitest";
import { centre, nodePosition, resort, terrainSize } from "../src/mountain/resort";

describe("the resort scene transform", () => {
    it("loads the Val-Tarin topology", () => {
        expect(resort.name).toBe("val-tarin");
        expect(resort.nodes).toHaveLength(60);
        expect(resort.edges).toHaveLength(80);
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
