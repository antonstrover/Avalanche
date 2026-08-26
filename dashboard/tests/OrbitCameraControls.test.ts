import { describe, expect, it } from "vitest";
import { Vector3 } from "three";
import { advanceCameraPose } from "../src/mountain/cameraMotion";

describe("OrbitCameraControls", () => {
    it("moves a smooth camera request to its exact goal", () => {
        const position = new Vector3(0, 0, 0);
        const target = new Vector3(0, 0, 0);
        const goal = {
            position: new Vector3(10, 20, 30),
            target: new Vector3(4, 5, 6),
        };

        expect(advanceCameraPose(position, target, goal)).toBe(false);
        expect(position.x).toBeGreaterThan(0);
        for (let frame = 0; frame < 100; frame += 1) {
            if (advanceCameraPose(position, target, goal)) break;
        }

        expect(position).toEqual(goal.position);
        expect(target).toEqual(goal.target);
    });
});
