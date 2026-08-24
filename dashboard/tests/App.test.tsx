import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../src/App";
import { mergeTimeline } from "../src/features/timeline";
import type { TimelineEvent } from "../src/workers/live-frame";
import resort from "../src/mountain/resort.json";

// The scene needs WebGL. The browser test covers it, so this test replaces it.
vi.mock("../src/mountain/MountainScene", () => ({
    MountainScene: () => null,
}));

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
                        scenarios: [{ id: "default", label: "Default" }],
                        controllers: [{ id: "honest", label: "Honest" }],
                        monitors: [{ id: "none", label: "None" }],
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
});
