import { Vector3 } from "three";
import { defaultResortModel, type ResortModel } from "./resort";
import type { Selection } from "./selection";
import type { HazardState } from "../workers/live-frame";

const MARKER_HEIGHT = 7;
const MARKER_SIZE = 2.2;

const colours: Record<string, string> = {
    low: "#f2d64b",
    medium: "#e8862a",
    high: "#d02f2f",
};

// The shape shows the severity, so the marker does not depend on colour alone.
function PrecursorShape({ severity }: { severity: HazardState["severity"] }) {
    if (severity === "high") {
        return <coneGeometry args={[MARKER_SIZE, MARKER_SIZE * 2, 3]} />;
    }
    if (severity === "medium") {
        return <octahedronGeometry args={[MARKER_SIZE]} />;
    }
    return <boxGeometry args={[MARKER_SIZE * 1.4, MARKER_SIZE * 1.4, MARKER_SIZE * 1.4]} />;
}

// A precursor on an edge marks the middle of that edge.
function precursorPosition(precursor: HazardState, model: ResortModel): Vector3 {
    const edge = model.resort.edges[precursor.edge_index];
    if (!edge) {
        throw new Error(`The precursor ${precursor.event_id} names an unknown edge.`);
    }
    const [source, destination] = model.edgeNodes(edge);
    return new Vector3().lerpVectors(
        model.nodePosition(source),
        model.nodePosition(destination),
        0.5,
    );
}

type Props = {
    hazards: HazardState[];
    selection: Selection;
    onSelect: (selection: Selection) => void;
    model?: ResortModel;
};

export function Hazards({
    hazards,
    selection,
    onSelect,
    model = defaultResortModel,
}: Props) {
    return (
        <group name="precursors">
            {hazards.map((precursor, index) => {
                const position = precursorPosition(precursor, model);
                const selected = selection?.kind === "hazard" && selection.index === index;
                return (
                    <mesh
                        key={precursor.event_id}
                        name={`precursor-${index}`}
                        position={[position.x, position.y + MARKER_HEIGHT, position.z]}
                        userData={{ eventType: precursor.event_type }}
                        onClick={(event) => {
                            event.stopPropagation();
                            onSelect({ kind: "hazard", index });
                        }}
                    >
                        <PrecursorShape severity={precursor.severity} />
                        <meshStandardMaterial
                            color={selected ? "#ffb020" : colours[precursor.severity]}
                            flatShading
                            roughness={0.5}
                        />
                    </mesh>
                );
            })}
        </group>
    );
}
