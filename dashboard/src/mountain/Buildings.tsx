import { defaultResortModel, type ResortModel } from "./resort";
import type { Selection } from "./selection";

const OFFSET = 4;

const colours: Record<string, string> = {
    entrance: "#d8c07a",
    exit: "#8f6ad0",
    lift_station: "#b8804a",
};

type Props = {
    selection: Selection;
    onSelect: (selection: Selection) => void;
    model?: ResortModel;
};

// A building is a simple box at an entrance, at an exit, or at a lift station.
export function Buildings({ selection, onSelect, model = defaultResortModel }: Props) {
    return (
        <group name="buildings">
            {model.buildingNodes.map((item) => {
                const position = model.nodePosition(item.node);
                const selected = selection?.kind === "building" && selection.index === item.index;
                const height = item.node.node_type === "lift_station" ? 4 : 3;
                return (
                    <mesh
                        key={item.index}
                        name={`building-${item.index}`}
                        position={[position.x + OFFSET, position.y + height / 2, position.z + OFFSET]}
                        onClick={(event) => {
                            event.stopPropagation();
                            onSelect({ kind: "building", index: item.index });
                        }}
                    >
                        <boxGeometry args={[5, height, 5]} />
                        <meshStandardMaterial
                            color={selected ? "#ffb020" : colours[item.node.node_type]}
                            flatShading
                        />
                    </mesh>
                );
            })}
        </group>
    );
}
