import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "../src/App";
import { mergeTimeline } from "../src/features/timeline";
import type { TimelineEvent } from "../src/workers/live-frame";

// The scene needs WebGL. The browser test covers it, so this test replaces it.
vi.mock("../src/mountain/MountainScene", () => ({
    MountainScene: () => null,
}));

describe("App shell", () => {
    beforeEach(() => {
        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue({
                json: () => Promise.resolve({ status: "ok" }),
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
