import { useMemo } from "react";
import { CatmullRomCurve3, Vector3 } from "three";
import { edgeNodes, lifts, nodePosition, type Edge } from "./resort";
import type { Selection } from "./selection";

const CABLE_HEIGHT = 5;
const GROUND_SAG = 2;
const PYLON_COUNT = 3;

type LiftShape = {
    cable: CatmullRomCurve3;
    pylons: { ground: Vector3; height: number }[];
    stations: Vector3[];
};

function liftShape(edge: Edge): LiftShape {
    const [source, destination] = edgeNodes(edge);
    const start = nodePosition(source);
    const end = nodePosition(destination);

    // The ground under the lift sags like the terrain. The cable stays above it.
    const ground = (fraction: number) => {
        const point = new Vector3().lerpVectors(start, end, fraction);
        point.y -= GROUND_SAG * Math.sin(Math.PI * fraction);
        return point;
    };
    const cablePoint = (fraction: number) => {
        const point = ground(fraction);
        point.y += CABLE_HEIGHT;
        return point;
    };

    const cable = new CatmullRomCurve3([cablePoint(0), cablePoint(0.5), cablePoint(1)]);
    const pylons = [];
    for (let index = 1; index <= PYLON_COUNT; index += 1) {
        const fraction = index / (PYLON_COUNT + 1);
        pylons.push({ ground: ground(fraction), height: CABLE_HEIGHT });
    }
    return { cable, pylons, stations: [start, end] };
}

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
