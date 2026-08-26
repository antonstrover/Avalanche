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
        if (!model.resort.nodes[reference.index]) return null;
        return { kind: "node", index: reference.index };
    }
    const edge = model.resort.edges[reference.index];
    if (!edge) return null;
    return {
        kind: edge.edge_type === "lift" ? "lift" : "piste",
        index: reference.index,
    };
}

export function referenceLabel(
    reference: InfrastructureReference,
    model: ResortModel,
): string {
    if (reference.kind === "node") {
        const node = model.resort.nodes[reference.index];
        return node ? node.node_id.replaceAll("_", " ") : `Node ${reference.index}`;
    }
    const edge = model.resort.edges[reference.index];
    return edge
        ? `${edge.source.replaceAll("_", " ")} to ${edge.destination.replaceAll("_", " ")}`
        : `Edge ${reference.index}`;
}
