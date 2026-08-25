import { Vector3 } from "three";
import { defaultResortModel, type ResortModel } from "./resort";
import { DIVERGENCE_COLOUR, divergentEdges } from "./telemetryView";
import type { AttackState, TelemetryState } from "../workers/live-frame";

const MARKER_HEIGHT = 11;
const MARKER_SIZE = 2.6;

// A divergent edge marks the middle of that edge.
function edgeMiddle(edge: number, model: ResortModel): Vector3 | null {
    const item = model.resort.edges[edge];
    if (!item) return null;
    const [source, destination] = model.edgeNodes(item);
    return new Vector3().lerpVectors(
        model.nodePosition(source),
        model.nodePosition(destination),
        0.5,
    );
}

// The ring shape carries the meaning, so the marker does not need its colour.
// The marker stays visible in the reported view and in the true view.
export function DivergenceMarkers({
    telemetry,
    attack,
    model = defaultResortModel,
}: {
    telemetry: TelemetryState;
    attack: AttackState;
    model?: ResortModel;
}) {
    const rows = divergentEdges(telemetry, attack);
    return (
        <group name="divergence">
            {rows.map((row) => {
                const position = edgeMiddle(row.edge, model);
                if (!position) return null;
                return (
                    <mesh
                        key={row.edge}
                        name={`divergence-${row.edge}`}
                        position={[position.x, position.y + MARKER_HEIGHT, position.z]}
                        rotation={[Math.PI / 2, 0, 0]}
                    >
                        <torusGeometry args={[MARKER_SIZE, MARKER_SIZE / 3, 6, 8]} />
                        <meshStandardMaterial
                            color={DIVERGENCE_COLOUR}
                            emissive={DIVERGENCE_COLOUR}
                            emissiveIntensity={0.5}
                            flatShading
                        />
                    </mesh>
                );
            })}
        </group>
    );
}
