import { useMemo } from "react";
import { CABLE_HEIGHT, liftShape } from "./curves";
import { lifts } from "./resort";
import type { Selection } from "./selection";

type Props = {
    selection: Selection;
    onSelect: (selection: Selection) => void;
};

export function Lifts({ selection, onSelect }: Props) {
    const shapes = useMemo(() => lifts.map((item) => liftShape(item.edge)), []);

    return (
        <group name="lifts">
            {lifts.map((item, order) => {
                const shape = shapes[order];
                const selected = selection?.kind === "lift" && selection.index === item.index;
                const colour = selected ? "#ffb020" : "#c8412f";
                return (
                    <group
                        key={item.index}
                        name={`lift-${item.index}`}
                        onClick={(event) => {
                            event.stopPropagation();
                            onSelect({ kind: "lift", index: item.index });
                        }}
                    >
                        <mesh>
                            <tubeGeometry args={[shape.cable, 16, 0.28, 6, false]} />
                            <meshStandardMaterial color={colour} flatShading />
                        </mesh>
                        {shape.pylons.map((pylon, pylonIndex) => (
                            <mesh
                                key={pylonIndex}
                                position={[pylon.ground.x, pylon.ground.y + pylon.height / 2, pylon.ground.z]}
                            >
                                <cylinderGeometry args={[0.3, 0.5, pylon.height, 5]} />
                                <meshStandardMaterial color={colour} flatShading />
                            </mesh>
                        ))}
                        {shape.stations.map((station, stationIndex) => (
                            <mesh
                                key={stationIndex}
                                position={[station.x, station.y + CABLE_HEIGHT / 2, station.z]}
                            >
                                <boxGeometry args={[4, CABLE_HEIGHT, 3]} />
                                <meshStandardMaterial color={colour} flatShading />
                            </mesh>
                        ))}
                    </group>
                );
            })}
        </group>
    );
}
