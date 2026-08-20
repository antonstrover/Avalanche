import { Vector3, type Curve } from "three";
import { liftShape, pisteCurve } from "./curves";
import { nodePosition, resort } from "./resort";

// A skier marker sits on the curve of its edge.
// The replay file gives the place of a skier: the kind, the index, and the progress.

export type Place = [kind: string, index: number, progress: number];

// One curve for each edge. A piste uses the piste curve. A lift uses the cable.
export const edgeCurves: Curve<Vector3>[] = resort.edges.map((edge) =>
    edge.edge_type === "lift" ? liftShape(edge).cable : pisteCurve(edge),
);

export const EDGE_POSITION_SAMPLES = 256;

export function workerPositionTable(): {
    nodePositions: Float32Array;
    edgePositions: Float32Array;
} {
    const nodePositions = new Float32Array(resort.nodes.length * 3);
    resort.nodes.forEach((node, index) => {
        const point = nodePosition(node);
        nodePositions.set([point.x, point.y, point.z], index * 3);
    });
    const edgePositions = new Float32Array(
        edgeCurves.length * EDGE_POSITION_SAMPLES * 3,
    );
    edgeCurves.forEach((curve, edge) => {
        for (let sample = 0; sample < EDGE_POSITION_SAMPLES; sample += 1) {
            const point = curve.getPointAt(sample / (EDGE_POSITION_SAMPLES - 1));
            const offset = (edge * EDGE_POSITION_SAMPLES + sample) * 3;
            edgePositions.set([point.x, point.y, point.z], offset);
        }
    });
    return { nodePositions, edgePositions };
}

// Return the point on one curve for a progress value from 0.0 to 1.0.
export function skierPosition(curve: Curve<Vector3>, progress: number): Vector3 {
    return curve.getPointAt(Math.min(Math.max(progress, 0), 1));
}

// Return the point of one replay place, or null when the scene draws no marker.
// A skier in a queue waits at the start of the lift.
export function placePosition(place: Place): Vector3 | null {
    const [kind, index, progress] = place;
    if (index < 0) return null;
    if (kind === "node") {
        const node = resort.nodes[index];
        return node ? nodePosition(node) : null;
    }
    const curve = edgeCurves[index];
    if (!curve) return null;
    return skierPosition(curve, kind === "queue" ? 0 : progress);
}
