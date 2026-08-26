import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApprovalPanel } from "../src/features/live/ApprovalPanel";
import type { LiveSession } from "../src/api/client";
import type { LiveAction, LiveDecision } from "../src/workers/live-frame";

const action: LiveAction = {
    route_weights: [[0]],
    piste_requests: [0],
    lift_capacity: [1],
    lift_capacity_enabled: [0],
    crowd_messages: [[0]],
    telemetry_overrides: [0],
    telemetry_override_enabled: [0],
};

const decision: LiveDecision = {
    proposal: {
        controller_id: "rule-demo",
        simulation_time: 0,
        action,
        explanation: "Close one evacuation route.",
        evidence: { target: "edge" },
    },
    executed_action: { controller_id: "pending-approval", simulation_time: 0, action },
    monitor_decision: {
        risk_score: 1,
        decision: "ESCALATE",
        reason_codes: ["EVACUATION_ROUTE_CLOSURE"],
        replacement_action: null,
        latency_seconds: 0.001,
        related_infrastructure: [],
    },
    fallback_source: "honest-fallback",
    predicted_result: { evacuation_score: 1 },
    approval: {
        decision_id: "0:rule-demo",
        status: "pending",
        choice: null,
        deadline_epoch_seconds: Date.now() / 1000 + 30,
        evidence: { target: "edge" },
        predicted_result: { evacuation_score: 1 },
        safe_fallback: action,
    },
};

const session: LiveSession = {
    session_id: "session-1",
    status: "running",
    skier_count: 20,
    simulation_speed: 20,
    frame_interval_ms: 250,
    topology_version: "topology",
    demo_failure: false,
    demo_monitor: false,
    demo_approval: true,
};

describe("ApprovalPanel", () => {
    afterEach(() => {
        cleanup();
        vi.unstubAllGlobals();
    });

    it("approves one pending escalation", async () => {
        vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true }));
        render(<ApprovalPanel decision={decision} session={session} />);

        expect(screen.getByTestId("approval-panel")).toBeInTheDocument();
        expect(screen.getByText("EVACUATION_ROUTE_CLOSURE")).toBeInTheDocument();
        fireEvent.click(screen.getByRole("button", { name: "Approve" }));

        await waitFor(() => {
            expect(screen.getByTestId("approval-status")).toHaveTextContent("accepted");
        });
    });

    it("stays hidden without a pending escalation", () => {
        render(<ApprovalPanel decision={null} session={session} />);
        expect(screen.queryByTestId("approval-panel")).not.toBeInTheDocument();
    });
});
