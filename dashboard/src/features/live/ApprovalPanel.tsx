import { useEffect, useState } from "react";
import { resolveApproval, type LiveSession } from "../../api/client";
import type { LiveDecision } from "../../workers/live-frame";

export function ApprovalPanel({
    decision,
    session,
}: {
    decision: LiveDecision | null;
    session: LiveSession | null;
}) {
    const approval = decision?.approval;
    const [now, setNow] = useState(0);
    const [replacement, setReplacement] = useState("");
    const [status, setStatus] = useState("");

    useEffect(() => {
        if (!approval || approval.status !== "pending") return;
        const firstTick = window.setTimeout(() => setNow(Date.now() / 1000), 0);
        const timer = window.setInterval(() => setNow(Date.now() / 1000), 250);
        return () => {
            window.clearTimeout(firstTick);
            window.clearInterval(timer);
        };
    }, [approval]);

    if (!approval || approval.status !== "pending" || !session) return null;
    const remaining = Math.max(approval.deadline_epoch_seconds - now, 0);
    const replacementValue = replacement || JSON.stringify(approval.safe_fallback, null, 2);

    const respond = async (choice: "APPROVE" | "BLOCK" | "REPLACE") => {
        setStatus("submitting");
        try {
            const action = choice === "REPLACE" ? JSON.parse(replacementValue) : undefined;
            await resolveApproval(session.session_id, approval.decision_id, choice, action);
            setStatus("accepted");
        } catch (error) {
            setStatus(error instanceof Error ? error.message : "the response failed");
        }
    };

    return (
        <section className="approval-panel" data-testid="approval-panel">
            <p className="eyebrow">Approval required</p>
            <h2>Resolve the escalated action</h2>
            <p data-testid="approval-deadline">{remaining.toFixed(1)} seconds remain</p>
            <h3>Proposed action</h3>
            <pre>{JSON.stringify(decision.proposal.action, null, 2)}</pre>
            <h3>Evidence</h3>
            <pre>{JSON.stringify(approval.evidence, null, 2)}</pre>
            <p>{decision.monitor_decision?.reason_codes.join(", ")}</p>
            <h3>Predicted result</h3>
            <pre>{JSON.stringify(approval.predicted_result, null, 2)}</pre>
            <h3>Safe fallback</h3>
            <textarea
                aria-label="Replacement action"
                value={replacementValue}
                onChange={(event) => setReplacement(event.target.value)}
                rows={10}
            />
            <div className="approval-actions">
                <button type="button" onClick={() => respond("APPROVE")}>Approve</button>
                <button type="button" onClick={() => respond("BLOCK")}>Block</button>
                <button type="button" onClick={() => respond("REPLACE")}>Replace</button>
            </div>
            <p aria-live="polite" data-testid="approval-status">{status}</p>
        </section>
    );
}
