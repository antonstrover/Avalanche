import { describe, expect, it, vi } from "vitest";
import data from "../src/mountain/resort.json";
import {
    cameraPresets,
    moveToPreset,
} from "../src/mountain/cameraPresets";
import { createResortModel, type Resort } from "../src/mountain/resort";

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
});
