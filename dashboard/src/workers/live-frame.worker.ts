/// <reference lib="webworker" />

import {
    decodeFrame,
    interpolatePositions,
    type FrameState,
    type PositionTable,
} from "./live-frame";

type Initialize = {
    type: "initialize";
    sessionId: string;
    topologyVersion: string;
    frameIntervalMs: number;
    nodePositions: ArrayBuffer;
    edgePositions: ArrayBuffer;
    edgeSamples: number;
    edgeCount: number;
};

type PackedFrame = { type: "frame"; packed: ArrayBuffer; receivedAt: number };
type Sample = { type: "sample"; now: number; buffer?: ArrayBuffer };

let sessionId = "";
let topologyVersion = "";
let frameIntervalMs = 250;
let table: PositionTable | null = null;
let before: FrameState | null = null;
let after: FrameState | null = null;
let recovering = false;

self.onmessage = (event: MessageEvent<Initialize | PackedFrame | Sample>) => {
    const message = event.data;
    try {
        if (message.type === "initialize") {
            sessionId = message.sessionId;
            topologyVersion = message.topologyVersion;
            frameIntervalMs = message.frameIntervalMs;
            table = {
                nodePositions: new Float32Array(message.nodePositions),
                edgePositions: new Float32Array(message.edgePositions),
                edgeSamples: message.edgeSamples,
                edgeCount: message.edgeCount,
            };
            return;
        }
        if (message.type === "frame") {
            const decoded = decodeFrame(
                message.packed,
                sessionId,
                topologyVersion,
                message.receivedAt,
            );
            if (decoded.type === "error") {
                self.postMessage({ type: "error", message: "the live session failed" });
                return;
            }
            if (!decoded.frame) return;
            if (
                decoded.type === "frame" &&
                after &&
                decoded.frame.sequence !== after.sequence + 1
            ) {
                recovering = true;
                self.postMessage({ type: "snapshot_request" });
                return;
            }
            if (decoded.type === "snapshot") {
                before = decoded.frame;
                after = decoded.frame;
                recovering = false;
            } else if (!recovering) {
                before = after ?? decoded.frame;
                after = decoded.frame;
            }
            self.postMessage({
                type: "accepted",
                sequence: decoded.frame.sequence,
                skierCount: decoded.frame.skierCount,
            });
            return;
        }
        if (message.type === "sample" && before && after && table && !recovering) {
            const positions = interpolatePositions(
                before,
                after,
                message.now,
                frameIntervalMs,
                table,
                message.buffer,
            );
            const visible = new Uint32Array(after.skierCount);
            let visibleCount = 0;
            for (let skier = 0; skier < after.skierCount; skier += 1) {
                if (Number.isFinite(positions[skier * 3])) {
                    visible[visibleCount] = skier;
                    visibleCount += 1;
                }
            }
            self.postMessage(
                {
                    type: "positions",
                    positions: positions.buffer,
                    visible: visible.buffer,
                    visibleCount,
                    count: after.skierCount,
                },
                { transfer: [positions.buffer, visible.buffer] },
            );
        }
    } catch (error) {
        self.postMessage({
            type: "error",
            message: error instanceof Error ? error.message : "the frame is invalid",
        });
    }
};
