import { CatmullRomCurve3, Vector3 } from "three";
import { defaultResortModel, type Edge, type ResortModel } from "./resort";

// The curve of one edge. The scene draws the edge on it.
// A skier marker also reads its position from it.

const SAG = 1.8;
const LIFT_OFF_GROUND = 0.7;
const PYLON_SPACING = 250;

// A piste is a curve between two nodes. The middle of the curve sags a little.
export function pisteCurve(
    edge: Edge,
    model: ResortModel = defaultResortModel,
): CatmullRomCurve3 {
    const [source, destination] = model.edgeNodes(edge);
    const start = model.nodePosition(source).setY(model.nodePosition(source).y + LIFT_OFF_GROUND);
    const end = model.nodePosition(destination).setY(model.nodePosition(destination).y + LIFT_OFF_GROUND);
    const middle = new Vector3().lerpVectors(start, end, 0.5);
    middle.y -= SAG;
    return new CatmullRomCurve3([start, middle, end], false, "catmullrom", 0.5);
}

export const CABLE_HEIGHT = 5;
const GROUND_SAG = 2;

export type LiftShape = {
    cable: CatmullRomCurve3;
    pylons: { ground: Vector3; height: number }[];
    stations: Vector3[];
};

export function liftShape(
    edge: Edge,
    model: ResortModel = defaultResortModel,
): LiftShape {
    const [source, destination] = model.edgeNodes(edge);
    const start = model.nodePosition(source);
    const end = model.nodePosition(destination);

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
    const pylonCount = Math.max(1, Math.ceil(edge.length / PYLON_SPACING) - 1);
    const pylons = [];
    for (let index = 1; index <= pylonCount; index += 1) {
        const fraction = index / (pylonCount + 1);
        pylons.push({ ground: ground(fraction), height: CABLE_HEIGHT });
    }
    return { cable, pylons, stations: [start, end] };
}
