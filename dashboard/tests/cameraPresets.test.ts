import { describe, expect, it, vi } from "vitest";
import data from "../src/mountain/resort.json";
import {
    cameraPresets,
    focusInfrastructure,
    moveToPreset,
} from "../src/mountain/cameraPresets";
import { createResortModel, type Resort } from "../src/mountain/resort";
import { Vector3 } from "three";

const model = createResortModel(data as Resort);

describe("cameraPresets", () => {
    it("derives three finite poses from the selected resort", () => {
        const presets = cameraPresets(model);

        expect(Object.keys(presets)).toEqual(["overview", "zone", "operations"]);
        for (const pose of Object.values(presets)) {
            expect([...pose.position, ...pose.target].every(Number.isFinite)).toBe(true);
        }
        expect(presets.zone.target).not.toEqual(presets.operations.target);
    });

    it("disables smooth movement when reduced motion is active", () => {
        const controls = { setLookAt: vi.fn() };
        const pose = cameraPresets(model).overview;

        moveToPreset(controls, pose, true);

        expect(controls.setLookAt).toHaveBeenCalledWith(
            ...pose.position,
            ...pose.target,
            false,
        );
    });

    it("focuses an edge midpoint and keeps the camera offset", () => {
        const controls = {
            setLookAt: vi.fn(),
            getPosition: (target: Vector3) => target.set(20, 30, 40),
            getTarget: (target: Vector3) => target.set(2, 3, 4),
        };
        const request = {
            id: 1,
            selection: { kind: "piste" as const, index: model.pistes[0].index },
        };

        expect(focusInfrastructure(controls, request, model, true)).toBe(true);

        const [x, y, z, targetX, targetY, targetZ, smooth] =
            controls.setLookAt.mock.calls[0];
        expect(x - targetX).toBeCloseTo(18);
        expect(y - targetY).toBeCloseTo(27);
        expect(z - targetZ).toBeCloseTo(36);
        expect([targetX, targetY, targetZ].every(Number.isFinite)).toBe(true);
        expect(smooth).toBe(false);
    });

    it("ignores an invalid focus index", () => {
        const controls = { setLookAt: vi.fn() };
        const request = {
            id: 1,
            selection: { kind: "node" as const, index: 1000 },
        };

        expect(focusInfrastructure(controls, request, model, false)).toBe(false);
        expect(controls.setLookAt).not.toHaveBeenCalled();
    });
});
