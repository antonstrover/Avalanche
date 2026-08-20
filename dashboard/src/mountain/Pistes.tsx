import { useMemo } from "react";
import { pisteCurve } from "./curves";
import { difficultyColour, pistes } from "./resort";
import type { Selection } from "./selection";

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
