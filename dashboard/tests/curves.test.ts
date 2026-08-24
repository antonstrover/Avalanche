import { describe, expect, it } from "vitest";
import { liftShape } from "../src/mountain/curves";
import { lifts } from "../src/mountain/resort";

describe("the lift shape", () => {
    it("gives a long lift more pylons than a short lift", () => {
        const ordered = lifts.toSorted((first, second) => first.edge.length - second.edge.length);
        const shortLift = ordered[0].edge;
        const longLift = ordered[ordered.length - 1].edge;

        expect(liftShape(longLift).pylons.length).toBeGreaterThan(
            liftShape(shortLift).pylons.length,
        );
    });
});
