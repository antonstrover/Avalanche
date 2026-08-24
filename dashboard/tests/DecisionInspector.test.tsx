import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { DecisionInspector } from "../src/features/live/DecisionInspector";
import type { LiveAction, LiveDecision, TelemetryState } from "../src/workers/live-frame";

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
    monitor_decision: {
        risk_score: 1,
        decision: "BLOCK",
        reason_codes: ["EVACUATION_ROUTE_CLOSURE"],
        replacement_action: null,
        latency_seconds: 0.001,
    },
    fallback_source: "honest-fallback",
    predicted_result: { evacuation_score: 1 },
};

const telemetry: TelemetryState = {
    reported_density: [0, 0], true_density: [0, 1],
    reported_occupancy: [0, 0], true_occupancy: [0, 1],
    reported_queue: [0, 0], true_queue: [0, 0],
    reported_speed: [1, 1], true_speed: [1, 0.5],
    reported_closed: [0, 0], true_closed: [0, 1],
};

describe("DecisionInspector", () => {
    afterEach(cleanup);

    it("shows the empty state before a live proposal", () => {
        render(<DecisionInspector decision={null} telemetry={telemetry} />);
        expect(screen.getByText("No proposal yet")).toBeInTheDocument();
    });

    it("shows an executed honest proposal", () => {
        render(<DecisionInspector decision={decision} telemetry={telemetry} />);
        expect(screen.getByTestId("proposal-controller")).toHaveTextContent("honest");
        expect(screen.getByTestId("proposal-explanation")).toHaveTextContent(
            "Reroute around closures.",
        );
        expect(screen.getByTestId("decision-type")).toHaveTextContent("BLOCK");
        expect(screen.getByTestId("reason-code")).toHaveTextContent(
            "EVACUATION_ROUTE_CLOSURE",
        );
        expect(screen.getByTestId("telemetry-comparison")).toBeInTheDocument();
    });
});
