import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
    LayerToggles,
} from "../src/mountain/MountainScene";
import { INITIAL_LAYERS, LAYER_NAMES } from "../src/mountain/layers";

afterEach(cleanup);

describe("LayerToggles", () => {
    it("starts with every named layer visible", () => {
        render(<LayerToggles layers={INITIAL_LAYERS} onChange={() => undefined} />);

        expect(LAYER_NAMES).toHaveLength(8);
        for (const name of LAYER_NAMES) {
            expect(screen.getByRole("checkbox", { name })).toBeChecked();
        }
    });

    it("changes only the selected layer", () => {
        const onChange = vi.fn();
        render(<LayerToggles layers={INITIAL_LAYERS} onChange={onChange} />);

        fireEvent.click(screen.getByRole("checkbox", { name: "hazards" }));

        expect(onChange).toHaveBeenCalledOnce();
        expect(onChange).toHaveBeenCalledWith("hazards", false);
    });
});
