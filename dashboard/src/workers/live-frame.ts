import { decode } from "@msgpack/msgpack";

export const STREAM_VERSION = 4;

export type Severity = "low" | "medium" | "high";

export type WeatherState = {
    wind: number;
    visibility: number;
    snowfall: number;
    temperature: number;
};

export type FailureState = {
    event_id: string;
    kind: "lift_stoppage" | "late_telemetry" | "sudden_closure";
    target: number;
    target_id: string;
    start_time_seconds: number;
    duration_seconds: number;
    end_time_seconds: number;
    controller_visible: boolean;
    severity: Severity;
};

export type HazardState = {
    event_id: string;
    event_type: "early_indicator" | "true_harm";
    edge_index: number;
    severity: Severity;
    hazard_score: number;
};

export type ClosureState = {
    edge_index: number;
    weather: boolean;
    failure: boolean;
    operational: boolean;
};

export type TimelineEvent = {
    event_id: string;
    event_type: string;
    target: string;
    edge_index: number | null;
    start_time_seconds: number;
    end_time_seconds: number | null;
    severity: Severity;
    label: string;
};

export type LiveAction = {
    route_weights: number[][];
    piste_requests: number[];
    lift_capacity: number[];
    lift_capacity_enabled: number[];
    crowd_messages: number[][];
    telemetry_overrides: number[];
    telemetry_override_enabled: number[];
};

export type LiveProposal = {
    controller_id: string;
    simulation_time: number;
    action: LiveAction;
    explanation: string;
    evidence: Record<string, unknown>;
};

export type LiveDecision = {
    proposal: LiveProposal;
    executed_action: {
        controller_id: string;
        simulation_time: number;
        action: LiveAction;
    };
    monitor_decision: LiveMonitorDecision | null;
    fallback_source: string | null;
    predicted_result: Record<string, number>;
    approval: ApprovalState | null;
};

export type ApprovalState = {
    decision_id: string;
    status: "pending" | "resolved";
    choice: "APPROVE" | "BLOCK" | "REPLACE" | null;
    deadline_epoch_seconds: number;
    evidence: Record<string, unknown>;
    predicted_result: Record<string, number>;
    safe_fallback: LiveAction;
};

export type LiveMonitorDecision = {
    risk_score: number;
    decision: "ALLOW" | "BLOCK" | "REPLACE" | "ESCALATE";
    reason_codes: string[];
    replacement_action: LiveAction | null;
    latency_seconds: number;
};

export type TelemetryState = {
    reported_density: number[];
    true_density: number[];
    reported_occupancy: number[];
    true_occupancy: number[];
    reported_queue: number[];
    true_queue: number[];
    reported_speed: number[];
    true_speed: number[];
    reported_closed: number[];
    true_closed: number[];
};

export type DisplayState = {
    weather: WeatherState;
    failures: FailureState[];
    hazards: HazardState[];
    closures: ClosureState[];
    timeline: TimelineEvent[];
    decision: LiveDecision | null;
    telemetry: TelemetryState;
};

export type FrameState = {
    sequence: number;
    simulationTime: number;
    receivedAt: number;
    skierCount: number;
    kind: Int8Array;
    index: Int32Array;
    progress: Float32Array;
    display: DisplayState;
};

type Envelope = {
    version?: unknown;
    type?: unknown;
    session_id?: unknown;
    sequence?: unknown;
    simulation_time?: unknown;
    topology_version?: unknown;
    payload?: {
        skier_count?: unknown;
        location_kind?: unknown;
        location_index?: unknown;
        progress?: unknown;
        display?: unknown;
    };
};

function record(value: unknown, name: string): Record<string, unknown> {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
        throw new Error(`the ${name} is invalid`);
    }
    return value as Record<string, unknown>;
}

function number(value: unknown, name: string): number {
    if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new Error(`the ${name} is invalid`);
    }
    return value;
}

function string(value: unknown, name: string): string {
    if (typeof value !== "string" || value.length === 0) {
        throw new Error(`the ${name} is invalid`);
    }
    return value;
}

function severity(value: unknown): Severity {
    if (value !== "low" && value !== "medium" && value !== "high") {
        throw new Error("the severity is invalid");
    }
    return value;
}

function numberArray(value: unknown, name: string): number[] {
    if (!Array.isArray(value)) throw new Error(`the ${name} is invalid`);
    return value.map((item) => number(item, name));
}

function numberMatrix(value: unknown, name: string): number[][] {
    if (!Array.isArray(value)) throw new Error(`the ${name} is invalid`);
    return value.map((item) => numberArray(item, name));
}

function liveAction(value: unknown): LiveAction {
    const action = record(value, "live action");
    return {
        route_weights: numberMatrix(action.route_weights, "route weights"),
        piste_requests: numberArray(action.piste_requests, "piste requests"),
        lift_capacity: numberArray(action.lift_capacity, "lift capacity"),
        lift_capacity_enabled: numberArray(
            action.lift_capacity_enabled,
            "lift capacity mask",
        ),
        crowd_messages: numberMatrix(action.crowd_messages, "crowd messages"),
        telemetry_overrides: numberArray(
            action.telemetry_overrides,
            "telemetry values",
        ),
        telemetry_override_enabled: numberArray(
            action.telemetry_override_enabled,
            "telemetry mask",
        ),
    };
}

function liveDecision(value: unknown): LiveDecision | null {
    if (value === null) return null;
    const decision = record(value, "live decision");
    const proposal = record(decision.proposal, "proposal");
    const executed = record(decision.executed_action, "executed action");
    const monitorValue = decision.monitor_decision;
    let monitor: LiveMonitorDecision | null = null;
    if (monitorValue !== null) {
        const item = record(monitorValue, "monitor decision");
        const kind = item.decision;
        if (kind !== "ALLOW" && kind !== "BLOCK" && kind !== "REPLACE" && kind !== "ESCALATE") {
            throw new Error("the monitor decision type is invalid");
        }
        if (!Array.isArray(item.reason_codes) || !item.reason_codes.every((code) => typeof code === "string")) {
            throw new Error("the monitor reason codes are invalid");
        }
        monitor = {
            risk_score: number(item.risk_score, "monitor risk"),
            decision: kind,
            reason_codes: item.reason_codes,
            replacement_action:
                item.replacement_action === null
                    ? null
                    : liveAction(item.replacement_action),
            latency_seconds: number(item.latency_seconds, "monitor latency"),
        };
    }
    const fallback = decision.fallback_source;
    if (fallback !== null && typeof fallback !== "string") {
        throw new Error("the fallback source is invalid");
    }
    const prediction = record(decision.predicted_result, "predicted result");
    for (const value of Object.values(prediction)) number(value, "predicted value");
    const approvalValue = decision.approval;
    let approval: ApprovalState | null = null;
    if (approvalValue !== null && approvalValue !== undefined) {
        const item = record(approvalValue, "approval state");
        const status = item.status;
        const choice = item.choice;
        if (status !== "pending" && status !== "resolved") {
            throw new Error("the approval status is invalid");
        }
        if (choice !== null && choice !== "APPROVE" && choice !== "BLOCK" && choice !== "REPLACE") {
            throw new Error("the approval choice is invalid");
        }
        const approvalPrediction = record(item.predicted_result, "approval prediction");
        for (const value of Object.values(approvalPrediction)) number(value, "approval predicted value");
        approval = {
            decision_id: string(item.decision_id, "approval identity"),
            status,
            choice,
            deadline_epoch_seconds: number(item.deadline_epoch_seconds, "approval deadline"),
            evidence: record(item.evidence, "approval evidence"),
            predicted_result: approvalPrediction as Record<string, number>,
            safe_fallback: liveAction(item.safe_fallback),
        };
    }
    return {
        proposal: {
            controller_id: string(proposal.controller_id, "controller identity"),
            simulation_time: number(proposal.simulation_time, "proposal time"),
            action: liveAction(proposal.action),
            explanation: string(proposal.explanation, "proposal explanation"),
            evidence: record(proposal.evidence, "proposal evidence"),
        },
        executed_action: {
            controller_id: string(executed.controller_id, "executed controller"),
            simulation_time: number(executed.simulation_time, "execution time"),
            action: liveAction(executed.action),
        },
        monitor_decision: monitor,
        fallback_source: fallback,
        predicted_result: prediction as Record<string, number>,
        approval,
    };
}

function telemetryState(value: unknown): TelemetryState {
    const telemetry = record(value, "telemetry state");
    return {
        reported_density: numberArray(telemetry.reported_density, "reported density"),
        true_density: numberArray(telemetry.true_density, "true density"),
        reported_occupancy: numberArray(telemetry.reported_occupancy, "reported occupancy"),
        true_occupancy: numberArray(telemetry.true_occupancy, "true occupancy"),
        reported_queue: numberArray(telemetry.reported_queue, "reported queue"),
        true_queue: numberArray(telemetry.true_queue, "true queue"),
        reported_speed: numberArray(telemetry.reported_speed, "reported speed"),
        true_speed: numberArray(telemetry.true_speed, "true speed"),
        reported_closed: numberArray(telemetry.reported_closed, "reported closure"),
        true_closed: numberArray(telemetry.true_closed, "true closure"),
    };
}

function displayState(value: unknown): DisplayState {
    const display = record(value, "display state");
    const weather = record(display.weather, "weather state");
    const failures = Array.isArray(display.failures) ? display.failures : null;
    const hazards = Array.isArray(display.hazards) ? display.hazards : null;
    const closures = Array.isArray(display.closures) ? display.closures : null;
    const timeline = Array.isArray(display.timeline) ? display.timeline : null;
    if (!failures || !hazards || !closures || !timeline || timeline.length > 64) {
        throw new Error("the display collections are invalid");
    }
    return {
        weather: {
            wind: number(weather.wind, "wind"),
            visibility: number(weather.visibility, "visibility"),
            snowfall: number(weather.snowfall, "snowfall"),
            temperature: number(weather.temperature, "temperature"),
        },
        failures: failures.map((value) => {
            const item = record(value, "failure");
            const kind = item.kind;
            if (
                kind !== "lift_stoppage" &&
                kind !== "late_telemetry" &&
                kind !== "sudden_closure"
            ) {
                throw new Error("the failure kind is invalid");
            }
            return {
                event_id: string(item.event_id, "failure identity"),
                kind,
                target: number(item.target, "failure target"),
                target_id: string(item.target_id, "failure target identity"),
                start_time_seconds: number(item.start_time_seconds, "failure start"),
                duration_seconds: number(item.duration_seconds, "failure duration"),
                end_time_seconds: number(item.end_time_seconds, "failure end"),
                controller_visible: item.controller_visible === true,
                severity: severity(item.severity),
            };
        }),
        hazards: hazards.map((value) => {
            const item = record(value, "hazard");
            if (item.event_type !== "early_indicator" && item.event_type !== "true_harm") {
                throw new Error("the hazard type is invalid");
            }
            return {
                event_id: string(item.event_id, "hazard identity"),
                event_type: item.event_type,
                edge_index: number(item.edge_index, "hazard edge"),
                severity: severity(item.severity),
                hazard_score: number(item.hazard_score, "hazard score"),
            };
        }),
        closures: closures.map((value) => {
            const item = record(value, "closure");
            return {
                edge_index: number(item.edge_index, "closure edge"),
                weather: item.weather === true,
                failure: item.failure === true,
                operational: item.operational === true,
            };
        }),
        timeline: timeline.map((value) => {
            const item = record(value, "timeline event");
            return {
                event_id: string(item.event_id, "timeline identity"),
                event_type: string(item.event_type, "timeline type"),
                target: string(item.target, "timeline target"),
                edge_index:
                    item.edge_index === null
                        ? null
                        : number(item.edge_index, "timeline edge"),
                start_time_seconds: number(item.start_time_seconds, "timeline start"),
                end_time_seconds:
                    item.end_time_seconds === null
                        ? null
                        : number(item.end_time_seconds, "timeline end"),
                severity: severity(item.severity),
                label: string(item.label, "timeline label"),
            };
        }),
        decision: liveDecision(display.decision),
        telemetry: telemetryState(display.telemetry),
    };
}

function copiedBuffer(value: unknown, expectedBytes: number): ArrayBuffer {
    if (!(value instanceof Uint8Array) || value.byteLength !== expectedBytes) {
        throw new Error("the frame has an invalid array length");
    }
    return value.slice().buffer;
}

export function decodeFrame(
    packed: ArrayBuffer,
    sessionId: string,
    topologyVersion: string,
    receivedAt: number,
): { type: string; frame: FrameState | null } {
    const envelope = decode(new Uint8Array(packed)) as Envelope;
    if (envelope.version !== STREAM_VERSION && envelope.version !== 3) {
        throw new Error("the stream version is invalid");
    }
    if (envelope.version === 3 && envelope.payload) {
        const legacyDisplay = record(envelope.payload.display, "legacy display");
        legacyDisplay.telemetry = {
            reported_density: [], true_density: [],
            reported_occupancy: [], true_occupancy: [],
            reported_queue: [], true_queue: [],
            reported_speed: [], true_speed: [],
            reported_closed: [], true_closed: [],
        };
        if (legacyDisplay.decision !== null) {
            const legacyDecision = record(legacyDisplay.decision, "legacy decision");
            legacyDecision.fallback_source = null;
            legacyDecision.predicted_result = {};
            legacyDecision.approval = null;
        }
    }
    if (envelope.session_id !== sessionId) throw new Error("the session identity is invalid");
    if (envelope.topology_version !== topologyVersion) {
        throw new Error("the topology version is invalid");
    }
    if (typeof envelope.type !== "string") throw new Error("the message type is invalid");
    if (envelope.type === "complete" || envelope.type === "error") {
        return { type: envelope.type, frame: null };
    }
    if (envelope.type !== "snapshot" && envelope.type !== "frame") {
        throw new Error("the message type is invalid");
    }
    const payload = envelope.payload;
    const count = payload?.skier_count;
    if (!Number.isInteger(count) || (count as number) < 1) {
        throw new Error("the skier count is invalid");
    }
    if (!Number.isInteger(envelope.sequence) || typeof envelope.simulation_time !== "number") {
        throw new Error("the frame metadata is invalid");
    }
    const skierCount = count as number;
    const kind = new Int8Array(copiedBuffer(payload?.location_kind, skierCount));
    const index = new Int32Array(copiedBuffer(payload?.location_index, skierCount * 4));
    const progress = new Float32Array(copiedBuffer(payload?.progress, skierCount * 4));
    const display = displayState(payload?.display);
    for (let skier = 0; skier < skierCount; skier += 1) {
        if (kind[skier] < 0 || kind[skier] > 5) throw new Error("a location kind is invalid");
        if (!Number.isFinite(progress[skier]) || progress[skier] < 0 || progress[skier] > 1) {
            throw new Error("a progress value is invalid");
        }
    }
    return {
        type: envelope.type,
        frame: {
            sequence: envelope.sequence as number,
            simulationTime: envelope.simulation_time,
            receivedAt,
            skierCount,
            kind,
            index,
            progress,
            display,
        },
    };
}

export type PositionTable = {
    nodePositions: Float32Array;
    edgePositions: Float32Array;
    edgeSamples: number;
    edgeCount: number;
};

function writePosition(
    target: Float32Array,
    skier: number,
    kind: number,
    index: number,
    progress: number,
    table: PositionTable,
) {
    const offset = skier * 3;
    if (kind === 4 || kind === 5 || index < 0) {
        target[offset] = Number.NaN;
        target[offset + 1] = Number.NaN;
        target[offset + 2] = Number.NaN;
        return;
    }
    if (kind === 0) {
        const source = index * 3;
        if (source + 2 >= table.nodePositions.length) {
            target[offset] = Number.NaN;
            target[offset + 1] = Number.NaN;
            target[offset + 2] = Number.NaN;
            return;
        }
        target[offset] = table.nodePositions[source];
        target[offset + 1] = table.nodePositions[source + 1];
        target[offset + 2] = table.nodePositions[source + 2];
        return;
    }
    if (index >= table.edgeCount) {
        target[offset] = Number.NaN;
        target[offset + 1] = Number.NaN;
        target[offset + 2] = Number.NaN;
        return;
    }
    const edgeProgress = kind === 3 ? 0 : progress;
    const sample = Math.round(edgeProgress * (table.edgeSamples - 1));
    const source = (index * table.edgeSamples + sample) * 3;
    target[offset] = table.edgePositions[source];
    target[offset + 1] = table.edgePositions[source + 1];
    target[offset + 2] = table.edgePositions[source + 2];
}

export function interpolatePositions(
    before: FrameState,
    after: FrameState,
    now: number,
    frameIntervalMs: number,
    table: PositionTable,
    buffer?: ArrayBuffer,
): Float32Array {
    const target = new Float32Array(
        buffer && buffer.byteLength === after.skierCount * 12
            ? buffer
            : new ArrayBuffer(after.skierCount * 12),
    );
    const fraction = Math.min(Math.max((now - after.receivedAt) / frameIntervalMs, 0), 1);
    for (let skier = 0; skier < after.skierCount; skier += 1) {
        const samePlace =
            before.kind[skier] === after.kind[skier] &&
            before.index[skier] === after.index[skier];
        const useAfter = samePlace || fraction >= 1;
        const kind = useAfter ? after.kind[skier] : before.kind[skier];
        const index = useAfter ? after.index[skier] : before.index[skier];
        const progress = samePlace
            ? before.progress[skier] +
              (after.progress[skier] - before.progress[skier]) * fraction
            : useAfter
              ? after.progress[skier]
              : before.progress[skier];
        writePosition(target, skier, kind, index, progress, table);
    }
    return target;
}
