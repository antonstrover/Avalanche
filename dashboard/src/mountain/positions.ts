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
