import { Vector3, type Curve } from "three";
import { liftShape, pisteCurve } from "./curves";
import { defaultResortModel, type ResortModel } from "./resort";

// A skier marker sits on the curve of its edge.
// The replay file gives the place of a skier: the kind, the index, and the progress.

export type Place = [kind: string, index: number, progress: number];

// One curve for each edge. A piste uses the piste curve. A lift uses the cable.
export function modelEdgeCurves(model: ResortModel): Curve<Vector3>[] {
    return model.resort.edges.map((edge) =>
        edge.edge_type === "lift"
            ? liftShape(edge, model).cable
            : pisteCurve(edge, model),
    );
}

export const edgeCurves = modelEdgeCurves(defaultResortModel);

export const EDGE_POSITION_SAMPLES = 256;

export function workerPositionTable(model: ResortModel = defaultResortModel): {
    nodePositions: Float32Array;
    edgePositions: Float32Array;
} {
    const curves = model === defaultResortModel ? edgeCurves : modelEdgeCurves(model);
    const nodePositions = new Float32Array(model.resort.nodes.length * 3);
    model.resort.nodes.forEach((node, index) => {
        const point = model.nodePosition(node);
        nodePositions.set([point.x, point.y, point.z], index * 3);
    });
    const edgePositions = new Float32Array(
        curves.length * EDGE_POSITION_SAMPLES * 3,
    );
    curves.forEach((curve, edge) => {
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
export function placePosition(
    place: Place,
    model: ResortModel = defaultResortModel,
    curves: Curve<Vector3>[] = edgeCurves,
): Vector3 | null {
    const [kind, index, progress] = place;
    if (index < 0) return null;
    if (kind === "node") {
        const node = model.resort.nodes[index];
        return node ? model.nodePosition(node) : null;
    }
    const curve = curves[index];
    if (!curve) return null;
    return skierPosition(curve, kind === "queue" ? 0 : progress);
}
