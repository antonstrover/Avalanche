import { Vector3 } from "three";
import { liftShape, pisteCurve } from "./curves";
import type { Selection } from "./selection";
import type { ResortModel } from "./resort";
import type {
    LiveAction,
    LiveDecision,
} from "../workers/live-frame";
import { referenceSelection, routeEdges } from "./routeOverlayState";

function edgeCurve(edgeIndex: number, model: ResortModel) {
    const edge = model.resort.edges[edgeIndex];
    if (!edge) return null;
    return edge.edge_type === "lift"
        ? liftShape(edge, model).cable
        : pisteCurve(edge, model);
}

function RouteLines({
    action,
    model,
    colour,
    radius,
    name,
}: {
    action: LiveAction;
    model: ResortModel;
    colour: string;
    radius: number;
    name: string;
}) {
    return routeEdges(action).map((edgeIndex) => {
        const curve = edgeCurve(edgeIndex, model);
        if (!curve) return null;
        return (
            <mesh key={`${name}-${edgeIndex}`} name={`${name}-route-${edgeIndex}`}>
                <tubeGeometry args={[curve, 24, radius, 8, false]} />
                <meshBasicMaterial color={colour} transparent opacity={0.86} />
            </mesh>
        );
    });
}

export function RouteOverlay({
    decision,
    model,
}: {
    decision: LiveDecision | null;
    model: ResortModel;
}) {
    if (!decision) return null;
    return (
        <group name="route-overlay">
            <RouteLines
                action={decision.proposal.action}
                model={model}
                colour="#2563eb"
                radius={1.15}
                name="proposed"
            />
            <RouteLines
                action={decision.executed_action.action}
                model={model}
                colour="#f59e0b"
                radius={0.62}
                name="executed"
            />
        </group>
    );
}

export function InterventionHighlights({
    decision,
    model,
    onSelect,
    onFocus,
}: {
    decision: LiveDecision | null;
    model: ResortModel;
    onSelect: (selection: Selection) => void;
    onFocus: () => void;
}) {
    const references = decision?.monitor_decision?.related_infrastructure ?? [];
    return references.map((reference) => {
        let position: Vector3 | null = null;
        if (reference.kind === "node") {
            const node = model.resort.nodes[reference.index];
            if (node) position = model.nodePosition(node);
        } else {
            const edge = model.resort.edges[reference.index];
            if (edge) {
                const [source, destination] = model.edgeNodes(edge);
                position = new Vector3().lerpVectors(
                    model.nodePosition(source),
                    model.nodePosition(destination),
                    0.5,
                );
            }
        }
        if (!position) return null;
        return (
            <mesh
                key={`${reference.kind}-${reference.index}`}
                name={`intervention-${reference.kind}-${reference.index}`}
                position={[position.x, position.y + 6, position.z]}
                onClick={(event) => {
                    event.stopPropagation();
                    onSelect(referenceSelection(reference, model));
                    onFocus();
                }}
            >
                <octahedronGeometry args={[3]} />
                <meshStandardMaterial color="#ef4444" emissive="#7f1d1d" />
            </mesh>
        );
    });
}
