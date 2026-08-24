import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { DecisionInspector } from "../src/features/live/DecisionInspector";
import type { LiveAction, LiveDecision } from "../src/workers/live-frame";

const action: LiveAction = {
    route_weights: [[1, 0]],
    piste_requests: [0, 2],
    lift_capacity: [1, 1],
    lift_capacity_enabled: [0, 0],
    crowd_messages: [[0]],
    telemetry_overrides: [0, 0],
    telemetry_override_enabled: [0, 0],
};

const decision: LiveDecision = {
    proposal: {
        controller_id: "honest",
        simulation_time: 60,
        action,
        explanation: "Reroute around closures.",
        evidence: { rules: ["reroute around closures"] },
    },
    executed_action: {
        controller_id: "honest",
        simulation_time: 60,
        action,
    },
    monitor_decision: null,
};

describe("DecisionInspector", () => {
    afterEach(cleanup);

    it("shows the empty state before a live proposal", () => {
        render(<DecisionInspector decision={null} />);
        expect(screen.getByText("No proposal yet")).toBeInTheDocument();
    });

    it("shows an executed honest proposal", () => {
        render(<DecisionInspector decision={decision} />);
        expect(screen.getByTestId("proposal-controller")).toHaveTextContent("honest");
        expect(screen.getByTestId("proposal-explanation")).toHaveTextContent(
            "Reroute around closures.",
        );
        expect(screen.getByText("Executed without a monitor")).toBeInTheDocument();
    });
});
