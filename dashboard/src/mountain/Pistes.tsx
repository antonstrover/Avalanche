import { useMemo } from "react";
import { pisteCurve } from "./curves";
import { difficultyColour, pistes } from "./resort";
import type { Selection } from "./selection";

type Props = {
    closedEdges: ReadonlySet<number>;
    selection: Selection;
    onSelect: (selection: Selection) => void;
};

export function Pistes({ closedEdges, selection, onSelect }: Props) {
    const curves = useMemo(() => pistes.map((item) => pisteCurve(item.edge)), []);

    return (
        <group name="pistes">
            {pistes.map((item, order) => {
                const selected = selection?.kind === "piste" && selection.index === item.index;
                const closed = closedEdges.has(item.index);
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
                            color={
                                selected
                                    ? "#ffb020"
                                    : closed
                                      ? "#6b2230"
                                      : difficultyColour[item.edge.difficulty]
                            }
                            opacity={closed ? 0.55 : 1}
                            transparent={closed}
                            flatShading
                            roughness={0.8}
                        />
                    </mesh>
                );
            })}
        </group>
    );
}
