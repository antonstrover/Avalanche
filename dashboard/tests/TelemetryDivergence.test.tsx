import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { TelemetryDivergence } from "../src/features/live/TelemetryDivergence";
import { NO_ATTACK, type AttackState, type TelemetryState } from "../src/workers/live-frame";

const telemetry: TelemetryState = {
    reported_density: [0.2, 0.4, 0.9],
    true_density: [0.2, 1.2, 0.9],
    reported_occupancy: [2, 4, 9],
    true_occupancy: [2, 12, 9],
    reported_queue: [0, 0, 0],
    true_queue: [0, 3, 0],
    reported_speed: [1, 1, 1],
    true_speed: [1, 0.4, 1],
    reported_closed: [0, 0, 0],
    true_closed: [0, 1, 0],
};

const hacker: AttackState = {
    kind: "reward_hacker",
    active: true,
    targets: [1, 2],
    divergent_edges: [1, 2],
};

afterEach(cleanup);

describe("TelemetryDivergence", () => {
    it("shows an empty state with no attack", () => {
        render(<TelemetryDivergence telemetry={telemetry} attack={NO_ATTACK} />);

        expect(screen.getByTestId("telemetry-divergence")).toHaveAttribute(
            "data-divergent-count",
            "0",
        );
        expect(screen.getByTestId("telemetry-divergence-live")).toHaveTextContent(
            "matches the true state",
        );
    });

    it("shows each divergent edge with both densities", () => {
        render(<TelemetryDivergence telemetry={telemetry} attack={hacker} />);

        expect(screen.getByTestId("telemetry-divergence")).toHaveAttribute(
            "data-divergent-count",
            "1",
        );
        expect(screen.getByTestId("divergent-reported-1")).toHaveTextContent("0.400");
        expect(screen.getByTestId("divergent-true-1")).toHaveTextContent("1.200");
        expect(screen.queryByTestId("divergent-edge-2")).toBeNull();
    });

    it("announces the divergence through a polite live region", () => {
        render(<TelemetryDivergence telemetry={telemetry} attack={hacker} />);

        expect(screen.getByTestId("telemetry-divergence-live")).toHaveAttribute(
            "aria-live",
            "polite",
        );
    });

    it("names the edge with the given label", () => {
        render(
            <TelemetryDivergence
                telemetry={telemetry}
                attack={hacker}
                edgeLabel={(edge) => `praz to plan ${edge}`}
            />,
        );

        expect(screen.getByTestId("divergent-edge-1")).toHaveTextContent(
            "praz to plan 1",
        );
    });
});
