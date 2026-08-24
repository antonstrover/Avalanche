import { useMemo } from "react";
import { pisteCurve } from "./curves";
import { defaultResortModel, difficultyColour, type ResortModel } from "./resort";
import type { Selection } from "./selection";
import { densityColour } from "./telemetryView";

type Props = {
    closedEdges: ReadonlySet<number>;
    density: readonly number[];
    selection: Selection;
    onSelect: (selection: Selection) => void;
    model?: ResortModel;
};

export function Pistes({
    closedEdges,
    density,
    selection,
    onSelect,
    model = defaultResortModel,
}: Props) {
    const curves = useMemo(
        () => model.pistes.map((item) => pisteCurve(item.edge, model)),
        [model],
    );

    return (
        <group name="pistes">
            {model.pistes.map((item, order) => {
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
                                      : densityColour(
                                            difficultyColour[item.edge.difficulty],
                                            density[item.index] ?? 0,
                                        )
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
