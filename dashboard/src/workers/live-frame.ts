import { decode } from "@msgpack/msgpack";

export const STREAM_VERSION = 1;

export type FrameState = {
    sequence: number;
    simulationTime: number;
    receivedAt: number;
    skierCount: number;
    kind: Int8Array;
    index: Int32Array;
    progress: Float32Array;
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
    };
};

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
    if (envelope.version !== STREAM_VERSION) throw new Error("the stream version is invalid");
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
