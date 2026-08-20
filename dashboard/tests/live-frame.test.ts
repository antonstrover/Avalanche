import { encode } from "@msgpack/msgpack";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import {
    decodeFrame,
    interpolatePositions,
    type DisplayState,
    type FrameState,
    type PositionTable,
} from "../src/workers/live-frame";

const display: DisplayState = {
    weather: { wind: 12, visibility: 700, snowfall: 4, temperature: -3 },
    failures: [],
    hazards: [],
    closures: [],
    timeline: [],
};

function packedFrame(sequence = 0): ArrayBuffer {
    const packed = encode({
        version: 2,
        type: sequence === 0 ? "snapshot" : "frame",
        session_id: "session-1",
        sequence,
        simulation_time: sequence * 5,
        topology_version: "topology-1",
        state_checksum: "checksum",
        payload: {
            skier_count: 2,
            location_kind: new Uint8Array([1, 5]),
            location_index: new Uint8Array(new Int32Array([0, 0]).buffer),
            progress: new Uint8Array(new Float32Array([0.5, 0]).buffer),
            display,
        },
    });
    return packed.buffer.slice(packed.byteOffset, packed.byteOffset + packed.byteLength) as ArrayBuffer;
}

describe("live frame handling", () => {
    it("decodes the shared stream contract fixture", () => {
        const encoded = readFileSync(
            "../tests/fixtures/live-frame-v2.msgpack.b64",
            "utf8",
        );
        const bytes = Buffer.from(encoded.trim(), "base64");
        const packed = bytes.buffer.slice(
            bytes.byteOffset,
            bytes.byteOffset + bytes.byteLength,
        ) as ArrayBuffer;
        const result = decodeFrame(
            packed,
            "fixture-session",
            "fixture-topology",
            100,
        );

        expect(result.frame?.sequence).toBe(3);
        expect(result.frame?.progress[0]).toBe(0.5);
        expect(result.frame?.display.weather.wind).toBe(12);
    });

    it("decodes the binary population arrays", () => {
        const result = decodeFrame(
            packedFrame(),
            "session-1",
            "topology-1",
            100,
        );
        expect(result.frame?.skierCount).toBe(2);
        expect(result.frame?.progress[0]).toBe(0.5);
        expect(result.frame?.kind[1]).toBe(5);
    });

    it("rejects an unexpected stream version", () => {
        const packed = encode({ version: 1, type: "frame" });
        const buffer = packed.buffer.slice(
            packed.byteOffset,
            packed.byteOffset + packed.byteLength,
        ) as ArrayBuffer;
        expect(() => decodeFrame(buffer, "session-1", "topology-1", 0)).toThrow(
            "the stream version is invalid",
        );
    });

    it("rejects an invalid binary array length", () => {
        const packed = encode({
            version: 2,
            type: "snapshot",
            session_id: "session-1",
            sequence: 0,
            simulation_time: 0,
            topology_version: "topology-1",
            payload: {
                skier_count: 2,
                location_kind: new Uint8Array([1]),
                location_index: new Uint8Array(8),
                progress: new Uint8Array(8),
                display,
            },
        });
        const buffer = packed.buffer.slice(
            packed.byteOffset,
            packed.byteOffset + packed.byteLength,
        ) as ArrayBuffer;

        expect(() => decodeFrame(buffer, "session-1", "topology-1", 0)).toThrow(
            "the frame has an invalid array length",
        );
    });

    it("interpolates one edge and hides a pending skier", () => {
        const frame = (progress: number, receivedAt: number): FrameState => ({
            sequence: receivedAt,
            simulationTime: receivedAt,
            receivedAt,
            skierCount: 2,
            kind: new Int8Array([1, 5]),
            index: new Int32Array([0, 0]),
            progress: new Float32Array([progress, 0]),
            display,
        });
        const table: PositionTable = {
            nodePositions: new Float32Array([0, 0, 0]),
            edgePositions: new Float32Array([0, 0, 0, 10, 0, 0, 20, 0, 0]),
            edgeSamples: 3,
            edgeCount: 1,
        };
        const positions = interpolatePositions(
            frame(0, 0),
            frame(1, 100),
            150,
            100,
            table,
        );
        expect(positions[0]).toBe(10);
        expect(positions[3]).toBeNaN();
    });

    it("holds the earlier place during a location change", () => {
        const frame = (index: number, receivedAt: number): FrameState => ({
            sequence: receivedAt,
            simulationTime: receivedAt,
            receivedAt,
            skierCount: 1,
            kind: new Int8Array([1]),
            index: new Int32Array([index]),
            progress: new Float32Array([0.5]),
            display,
        });
        const table: PositionTable = {
            nodePositions: new Float32Array(0),
            edgePositions: new Float32Array([0, 0, 0, 10, 0, 0, 20, 0, 0]),
            edgeSamples: 1,
            edgeCount: 3,
        };

        const positions = interpolatePositions(
            frame(0, 0),
            frame(1, 100),
            150,
            100,
            table,
        );
        expect(positions[0]).toBe(0);
    });
});
