import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../src/App";
import { mergeTimeline } from "../src/features/timeline";
import type { TimelineEvent } from "../src/workers/live-frame";
import resort from "../src/mountain/resort.json";

// The scene needs WebGL. The browser test covers it, so this test replaces it.
vi.mock("../src/mountain/MountainScene", async () => {
    const { INITIAL_DISPLAY } = await import("../src/mountain/conditions");
    return {
        MountainScene: ({
            onLiveFrame,
            selection,
            onSelectionChange,
        }: {
            onLiveFrame: (count: number, display: typeof INITIAL_DISPLAY) => void;
            selection: { kind: string; index: number } | null;
            onSelectionChange: (
                selection: { kind: "building"; index: number },
            ) => void;
        }) => (
            <div>
                <button
                    type="button"
                    onClick={() => onLiveFrame(5000, INITIAL_DISPLAY)}
                >
                    Emit a live frame
                </button>
                <button
                    type="button"
                    onClick={() => onSelectionChange({ kind: "building", index: 2 })}
                >
                    Select a building
                </button>
                <p data-testid="scene-selection">
                    {selection ? `${selection.kind}: ${selection.index}` : "none"}
                </p>
            </div>
        ),
    };
});

const liveSession = {
    session_id: "session-one",
    status: "running",
    skier_count: 5000,
    simulation_speed: 20,
    frame_interval_ms: 250,
    topology_version: "one",
    demo_failure: false,
    demo_monitor: false,
    demo_approval: false,
    resolved_config: {},
};

describe("App shell", () => {
    beforeEach(() => {
        vi.stubGlobal(
            "fetch",
            vi.fn((input: string | URL | Request) => {
                const url = String(input);
                let body: object = { status: "ok" };
                if (url.endsWith("/api/config-options")) {
                    body = {
                        mountains: [
                            {
                                id: "medium-resort",
                                label: "Val Tarin",
                                topology: resort,
                            },
                        ],
                        scenarios: [
                            {
                                id: "default",
                                label: "Default",
                                compatible_mountain_ids: ["medium-resort"],
                            },
                        ],
                        controllers: [
                            {
                                id: "honest",
                                label: "Honest",
                                compatible_mountain_ids: ["medium-resort"],
                                controller: { kind: "honest" },
                            },
                        ],
                        monitors: [
                            {
                                id: "none",
                                label: "None",
                                compatible_mountain_ids: ["medium-resort"],
                            },
                        ],
                    };
                } else if (url.endsWith("/api/config-options/resolve")) {
                    body = {
                        mountain: { name: "val-tarin" },
                        population: { skier_count: 5000 },
                        scenario: { name: "default" },
                        controller: { kind: "honest" },
                        monitor: { kind: "none" },
                        seed: 0,
                    };
                } else if (url.endsWith("/api/sessions")) {
                    body = liveSession;
                } else if (url.endsWith("/commands")) {
                    body = { ...liveSession, status: "paused" };
                }
                return Promise.resolve({
                    ok: true,
                    json: () => Promise.resolve(body),
                });
            }),
        );
    });

    afterEach(() => {
        cleanup();
        vi.unstubAllGlobals();
    });

    it("renders the fetched health status", async () => {
        render(<App />);

        await waitFor(() => {
            expect(screen.getByTestId("health-status")).toHaveTextContent(
                "API status: ok",
            );
        });
    });

    it("shows the resort name and its topology counts", () => {
        render(<App />);

        expect(screen.getByTestId("resort-name")).toHaveTextContent(
            "Val-Tarin · 60 nodes · 80 edges",
        );
    });

    it("shows a compact resolved configuration", async () => {
        render(<App />);

        await waitFor(() => {
            expect(screen.getByTestId("resolved-config")).toHaveTextContent(
                "val-tarin",
            );
        });
        expect(screen.getByTestId("resolved-config")).toHaveTextContent("5000");
        expect(
            screen.getByText("View the full configuration").closest("details"),
        ).not.toHaveAttribute("open");
        expect(fetch).toHaveBeenCalledWith("/api/config-options");
    });

    it("deduplicates recovered timeline events by their stable identity", () => {
        const event: TimelineEvent = {
            event_id: "failure:one:start",
            event_type: "failure_started",
            target: "lift one",
            edge_index: 1,
            start_time_seconds: 5,
            end_time_seconds: 65,
            severity: "high",
            label: "lift stoppage",
        };

        expect(mergeTimeline([event], [event])).toEqual([event]);
    });

    it("keeps the session paused when a queued frame arrives", async () => {
        render(<App />);

        const start = screen.getByRole("button", { name: "Start live session" });
        await waitFor(() => expect(start).toBeEnabled());
        fireEvent.click(start);
        fireEvent.click(await screen.findByRole("button", { name: "Emit a live frame" }));
        await waitFor(() => expect(screen.getByRole("button", { name: "Pause" })).toBeEnabled());

        fireEvent.click(screen.getByRole("button", { name: "Pause" }));
        await waitFor(() => expect(screen.getByTestId("live-status")).toHaveTextContent("paused"));
        fireEvent.click(screen.getByRole("button", { name: "Emit a live frame" }));

        expect(screen.getByTestId("live-status")).toHaveTextContent("paused");
        expect(screen.getByRole("button", { name: "Resume" })).toBeEnabled();
        expect(screen.getByRole("button", { name: "Pause" })).toBeDisabled();
    });

    it("keeps one shared mountain selection", () => {
        render(<App />);

        fireEvent.click(screen.getByRole("button", { name: "Select a building" }));

        expect(screen.getByTestId("scene-selection")).toHaveTextContent("building: 2");
    });
});
