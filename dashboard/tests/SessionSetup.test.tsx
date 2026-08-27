import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ConfigOptionsResponse } from "../src/api/client";
import { SessionSetup } from "../src/features/live/SessionSetup";

const options = {
    mountains: [
        { id: "medium-resort", label: "Val Tarin" },
        { id: "small-resort", label: "Small Resort" },
    ],
    scenarios: [
        {
            id: "default",
            label: "Default",
            compatible_mountain_ids: ["medium-resort", "small-resort"],
        },
    ],
    controllers: [
        {
            id: "honest",
            label: "Honest",
            compatible_mountain_ids: ["medium-resort"],
            controller: { kind: "honest" },
        },
        {
            id: "none",
            label: "None",
            compatible_mountain_ids: ["medium-resort", "small-resort"],
            controller: { kind: "none" },
        },
        {
            id: "small-resort/honest",
            label: "Honest",
            compatible_mountain_ids: ["small-resort"],
            controller: { kind: "honest" },
        },
    ],
    monitors: [
        {
            id: "none",
            label: "None",
            compatible_mountain_ids: ["medium-resort", "small-resort"],
        },
    ],
} as unknown as ConfigOptionsResponse;

const selection = {
    mountain: "medium-resort",
    scenario: "default",
    controller: "honest",
    monitor: "none",
    seed: 0,
    population: { skier_count: 20 },
    frame_interval_ms: 250,
    simulation_speed: 20,
};

describe("SessionSetup", () => {
    afterEach(cleanup);

    it("offers only controllers for the selected mountain", () => {
        render(
            <SessionSetup
                options={options}
                selection={selection}
                resolved={null}
                failed={false}
                onChange={() => undefined}
            />,
        );

        const controller = screen.getByRole("combobox", { name: "controller" });
        expect(controller).toHaveTextContent("Honest");
        expect(controller).toHaveTextContent("None");
        expect(controller.querySelectorAll("option")).toHaveLength(2);
    });

    it("changes a mountain and its controller together", () => {
        const onChange = vi.fn();
        render(
            <SessionSetup
                options={options}
                selection={selection}
                resolved={null}
                failed={false}
                onChange={onChange}
            />,
        );

        fireEvent.change(screen.getByRole("combobox", { name: "mountain" }), {
            target: { value: "small-resort" },
        });

        expect(onChange).toHaveBeenCalledOnce();
        expect(onChange).toHaveBeenCalledWith({
            ...selection,
            mountain: "small-resort",
            controller: "small-resort/honest",
        });
    });
});
