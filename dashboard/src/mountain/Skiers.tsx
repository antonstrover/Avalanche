import { useEffect, useMemo, useRef, useState } from "react";
import { useFrame } from "@react-three/fiber";
import { encode } from "@msgpack/msgpack";
import { Matrix4, Vector3, type InstancedMesh } from "three";
import { liveStreamUrl, type LiveSession } from "../api/client";
import { reducedMotion } from "./conditions";
import {
    EDGE_POSITION_SAMPLES,
    modelEdgeCurves,
    placePosition,
    workerPositionTable,
    type Place,
} from "./positions";
import data from "./replay.sample.json";
import type { DisplayState } from "../workers/live-frame";
import { defaultResortModel, type ResortModel } from "./resort";

type Frame = { time: number; skiers: Place[] };
type Replay = { skier_count: number; frames: Frame[] };

const replay = data as unknown as Replay;
const MARKER_RADIUS = 1;
const MARKER_HEIGHT = 3;
const SPEED = 20;
const STILL_FRAME = 24;
const HIDDEN = new Matrix4().makeScale(0, 0, 0);

function frameTime(index: number): number {
    return replay.frames[index].time;
}

type WorkerMessage = {
    type: string;
    positions?: ArrayBuffer;
    visible?: ArrayBuffer;
    visibleCount?: number;
    count?: number;
    skierCount?: number;
    display?: DisplayState;
};

export function Skiers({
    session,
    onLiveFrame,
    onLiveError,
    model = defaultResortModel,
}: {
    session: LiveSession | null;
    onLiveFrame: (count: number, display: DisplayState) => void;
    onLiveError: () => void;
    model?: ResortModel;
}) {
    const mesh = useRef<InstancedMesh>(null);
    const worker = useRef<Worker | null>(null);
    const positions = useRef<Float32Array | null>(null);
    const visible = useRef<Uint32Array | null>(null);
    const visibleCount = useRef(0);
    const positionsDirty = useRef(false);
    const recycled = useRef<ArrayBuffer | undefined>(undefined);
    const samplePending = useRef(false);
    const liveReady = useRef(false);
    const liveMatricesReady = useRef(false);
    const [skierCount, setSkierCount] = useState(replay.skier_count);
    const still = useMemo(() => reducedMotion(), []);
    const time = useRef(frameTime(0));
    const cursor = useRef(0);
    const matrix = useMemo(() => new Matrix4(), []);
    const point = useMemo(() => new Vector3(), []);
    const curves = useMemo(() => modelEdgeCurves(model), [model]);
    const sessionId = session?.session_id;
    const topologyVersion = session?.topology_version;
    const frameIntervalMs = session?.frame_interval_ms;

    useEffect(() => {
        if (!sessionId || !topologyVersion || frameIntervalMs === undefined) return;
        const frameWorker = new Worker(
            new URL("../workers/live-frame.worker.ts", import.meta.url),
            { type: "module" },
        );
        worker.current = frameWorker;
        const table = workerPositionTable(model);
        frameWorker.postMessage(
            {
                type: "initialize",
                sessionId,
                topologyVersion,
                frameIntervalMs: still ? 1 : frameIntervalMs,
                nodePositions: table.nodePositions.buffer,
                edgePositions: table.edgePositions.buffer,
                edgeSamples: EDGE_POSITION_SAMPLES,
                edgeCount: curves.length,
            },
            [table.nodePositions.buffer, table.edgePositions.buffer],
        );

        const socket = new WebSocket(liveStreamUrl(sessionId));
        socket.binaryType = "arraybuffer";
        socket.onmessage = (event: MessageEvent<ArrayBuffer>) => {
            frameWorker.postMessage(
                { type: "frame", packed: event.data, receivedAt: performance.now() },
                [event.data],
            );
        };
        socket.onerror = onLiveError;
        socket.onclose = (event) => {
            if (event.code !== 1000) onLiveError();
        };

        frameWorker.onmessage = (event: MessageEvent<WorkerMessage>) => {
            const message = event.data;
            if (message.type === "snapshot_request" && socket.readyState === WebSocket.OPEN) {
                socket.send(encode({ version: 3, type: "snapshot_request" }));
            } else if (
                message.type === "accepted" &&
                message.skierCount &&
                message.display
            ) {
                liveReady.current = true;
                setSkierCount(message.skierCount);
                onLiveFrame(message.skierCount, message.display);
            } else if (
                message.type === "positions" &&
                message.positions &&
                message.visible &&
                message.visibleCount !== undefined
            ) {
                if (positions.current) {
                    recycled.current = positions.current.buffer as ArrayBuffer;
                }
                positions.current = new Float32Array(message.positions);
                visible.current = new Uint32Array(message.visible);
                visibleCount.current = message.visibleCount;
                positionsDirty.current = true;
                samplePending.current = false;
            } else if (message.type === "error") {
                onLiveError();
            }
        };

        return () => {
            socket.close(1000);
            frameWorker.terminate();
            worker.current = null;
            positions.current = null;
            visible.current = null;
            visibleCount.current = 0;
            positionsDirty.current = false;
            samplePending.current = false;
            liveReady.current = false;
            liveMatricesReady.current = false;
            setSkierCount(replay.skier_count);
        };
    }, [
        sessionId,
        topologyVersion,
        frameIntervalMs,
        onLiveError,
        onLiveFrame,
        still,
        model,
        curves,
    ]);

    useFrame((_state, delta) => {
        const instances = mesh.current;
        if (!instances) return;

        if (session && worker.current) {
            const livePositions = positions.current;
            const liveVisible = visible.current;
            if (livePositions && liveVisible && positionsDirty.current) {
                const values = instances.instanceMatrix.array as Float32Array;
                if (!liveMatricesReady.current) {
                    values.fill(0);
                    for (let order = 0; order < skierCount; order += 1) {
                        values[order * 16 + 15] = 1;
                    }
                    liveMatricesReady.current = true;
                }
                for (let order = 0; order < visibleCount.current; order += 1) {
                    const skier = liveVisible[order];
                    const positionOffset = skier * 3;
                    const matrixOffset = order * 16;
                    const x = livePositions[positionOffset];
                    values[matrixOffset] = 1;
                    values[matrixOffset + 5] = 1;
                    values[matrixOffset + 10] = 1;
                    values[matrixOffset + 12] = x;
                    values[matrixOffset + 13] =
                        livePositions[positionOffset + 1] + MARKER_HEIGHT / 2;
                    values[matrixOffset + 14] = livePositions[positionOffset + 2];
                }
                instances.count = visibleCount.current;
                positionsDirty.current = false;
                instances.instanceMatrix.needsUpdate = true;
            }
            if (liveReady.current && !samplePending.current) {
                const buffer = recycled.current;
                recycled.current = undefined;
                samplePending.current = true;
                worker.current.postMessage(
                    { type: "sample", now: performance.now(), buffer },
                    buffer ? [buffer] : [],
                );
            }
            return;
        }

        const first = frameTime(0);
        const last = frameTime(replay.frames.length - 1);
        if (still) {
            time.current = frameTime(STILL_FRAME);
        } else {
            time.current += delta * SPEED;
            if (time.current > last) time.current = first;
        }
        if (frameTime(cursor.current) > time.current) cursor.current = 0;
        while (
            cursor.current + 1 < replay.frames.length &&
            frameTime(cursor.current + 1) <= time.current
        ) {
            cursor.current += 1;
        }
        const before = replay.frames[cursor.current];
        const after = replay.frames[cursor.current + 1] ?? before;
        const span = after.time - before.time;
        const fraction = span > 0 ? (time.current - before.time) / span : 0;
        for (let skier = 0; skier < replay.skier_count; skier += 1) {
            const start = before.skiers[skier];
            const end = after.skiers[skier];
            const startPoint = start ? placePosition(start, model, curves) : null;
            const endPoint = end ? placePosition(end, model, curves) : null;
            const sameEdge = start && end && start[0] === end[0] && start[1] === end[1];
            const place =
                startPoint && endPoint && sameEdge
                    ? point.copy(startPoint).lerp(endPoint, fraction)
                    : (endPoint ?? startPoint);
            if (!place) {
                instances.setMatrixAt(skier, HIDDEN);
                continue;
            }
            matrix.makeTranslation(place.x, place.y + MARKER_HEIGHT / 2, place.z);
            instances.setMatrixAt(skier, matrix);
        }
        instances.instanceMatrix.needsUpdate = true;
    });

    return (
        <instancedMesh
            ref={mesh}
            name="skiers"
            key={session?.session_id ?? "replay"}
            args={[undefined, undefined, skierCount]}
            frustumCulled={false}
            raycast={() => null}
        >
            <coneGeometry args={[MARKER_RADIUS, MARKER_HEIGHT, 3]} />
            <meshBasicMaterial color="#1b2233" />
        </instancedMesh>
    );
}
