import { useMemo } from "react";
import { CatmullRomCurve3, Vector3 } from "three";
import { difficultyColour, edgeNodes, nodePosition, pistes, type Edge } from "./resort";
import type { Selection } from "./selection";

const SAG = 1.8;
const LIFT_OFF_GROUND = 0.7;

// A piste is a curve between two nodes. The middle of the curve sags a little.
function pisteCurve(edge: Edge): CatmullRomCurve3 {
    const [source, destination] = edgeNodes(edge);
    const start = nodePosition(source).setY(nodePosition(source).y + LIFT_OFF_GROUND);
    const end = nodePosition(destination).setY(nodePosition(destination).y + LIFT_OFF_GROUND);
    const middle = new Vector3().lerpVectors(start, end, 0.5);
    middle.y -= SAG;
    return new CatmullRomCurve3([start, middle, end], false, "catmullrom", 0.5);
}

type Props = {
    selection: Selection;
    onSelect: (selection: Selection) => void;
};

export function Pistes({ selection, onSelect }: Props) {
    const curves = useMemo(() => pistes.map((item) => pisteCurve(item.edge)), []);

    return (
        <group name="pistes">
            {pistes.map((item, order) => {
                const selected = selection?.kind === "piste" && selection.index === item.index;
                return (
                    <mesh
                        key={item.index}
                        name={`piste-${item.index}`}
                        onClick={(event) => {
                            event.stopPropagation();
                            onSelect({ kind: "piste", index: item.index });
                        }}
                    >
                        <tubeGeometry args={[curves[order], 24, selected ? 1.6 : 1.2, 8, false]} />
                        <meshStandardMaterial
                            color={selected ? "#ffb020" : difficultyColour[item.edge.difficulty]}
                            flatShading
                            roughness={0.8}
                        />
                    </mesh>
                );
            })}
        </group>
    );
}
