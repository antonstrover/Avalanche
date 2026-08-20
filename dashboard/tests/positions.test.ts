import { CatmullRomCurve3, Vector3 } from "three";
import { describe, expect, it } from "vitest";
import { placePosition, skierPosition } from "../src/mountain/positions";
import { nodePosition, resort } from "../src/mountain/resort";

// A straight curve with a known length. The progress then maps to a known point.
const curve = new CatmullRomCurve3([
    new Vector3(0, 0, 0),
    new Vector3(5, 0, 0),
    new Vector3(10, 0, 0),
]);

function near(point: Vector3, x: number, y: number, z: number) {
    expect(point.x).toBeCloseTo(x, 3);
    expect(point.y).toBeCloseTo(y, 3);
    expect(point.z).toBeCloseTo(z, 3);
}

describe("the marker position", () => {
    it("gives the start of the curve for the progress 0", () => {
        near(skierPosition(curve, 0), 0, 0, 0);
    });

    it("gives the middle of the curve for the progress 0.5", () => {
        near(skierPosition(curve, 0.5), 5, 0, 0);
    });

    it("gives the end of the curve for the progress 1", () => {
        near(skierPosition(curve, 1), 10, 0, 0);
    });

    it("holds a progress value outside the range inside the curve", () => {
        near(skierPosition(curve, -2), 0, 0, 0);
        near(skierPosition(curve, 4), 10, 0, 0);
    });
});

describe("the place of one replay entry", () => {
    it("moves along the edge with the progress value", () => {
        const start = placePosition(["piste", 0, 0]);
        const middle = placePosition(["piste", 0, 0.5]);
        const end = placePosition(["piste", 0, 1]);
        expect(start).not.toBeNull();
        expect(start!.distanceTo(middle!)).toBeGreaterThan(0);
        expect(start!.distanceTo(end!)).toBeGreaterThan(start!.distanceTo(middle!));
    });

    it("puts a skier in a queue at the start of the edge", () => {
        const queue = placePosition(["queue", 1, 0.8]);
        const start = placePosition(["lift", 1, 0]);
        near(queue!, start!.x, start!.y, start!.z);
    });

    it("puts a skier at a node on that node", () => {
        const place = placePosition(["node", 0, 0]);
        const node = nodePosition(resort.nodes[0]);
        near(place!, node.x, node.y, node.z);
    });

    it("draws no marker for a skier that finished", () => {
        expect(placePosition(["finished", -1, 0])).toBeNull();
    });
});
