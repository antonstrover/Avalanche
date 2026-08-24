import { Vector3 } from "three";
import { defaultResortModel, type ResortModel } from "./resort";
import type { FailureState } from "../workers/live-frame";

function failurePosition(
    failure: FailureState,
    model: ResortModel,
): Vector3 | null {
    const edge = model.resort.edges[failure.target];
    if (!edge) return null;
    const [source, destination] = model.edgeNodes(edge);
    return new Vector3().lerpVectors(
        model.nodePosition(source),
        model.nodePosition(destination),
        0.5,
    );
}

export function Failures({
    failures,
    model = defaultResortModel,
}: {
    failures: FailureState[];
    model?: ResortModel;
}) {
    return (
        <group name="failures">
            {failures.map((failure) => {
                const position = failurePosition(failure, model);
                if (!position) return null;
                return (
                    <mesh
                        key={failure.event_id}
                        name={`failure-${failure.target}`}
                        position={[position.x, position.y + 10, position.z]}
                    >
                        <torusGeometry args={[3.2, 0.9, 6, 12]} />
                        <meshStandardMaterial color="#7c3aed" emissive="#3b1678" />
                    </mesh>
                );
            })}
        </group>
    );
}
