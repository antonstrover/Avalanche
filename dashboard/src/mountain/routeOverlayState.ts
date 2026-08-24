import type { Selection } from "./selection";
import type { ResortModel } from "./resort";
import type {
    InfrastructureReference,
    LiveAction,
} from "../workers/live-frame";

export function routeEdges(action: LiveAction): number[] {
    const edgeCount = action.route_weights[0]?.length ?? 0;
    const edges = [];
    for (let edge = 0; edge < edgeCount; edge += 1) {
        if (action.route_weights.some((row) => Math.abs(row[edge] ?? 0) > 0)) {
            edges.push(edge);
        }
    }
    return edges;
}

export function referenceSelection(
    reference: InfrastructureReference,
    model: ResortModel,
): Selection {
    if (reference.kind === "node") {
        return { kind: "node", index: reference.index };
    }
    const edge = model.resort.edges[reference.index];
    return {
        kind: edge?.edge_type === "lift" ? "lift" : "piste",
        index: reference.index,
    };
}
