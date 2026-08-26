import { Vector3 } from "three";
import { liftShape, pisteCurve } from "./curves";
import type { ResortModel } from "./resort";
import type { FocusRequest } from "./selection";

export type CameraPresetName = "overview" | "zone" | "operations";

export type CameraPose = {
    position: [number, number, number];
    target: [number, number, number];
};

export type CameraPresetMap = Record<CameraPresetName, CameraPose>;

export type CameraControl = {
    setLookAt: (
        x: number,
        y: number,
        z: number,
        targetX: number,
        targetY: number,
        targetZ: number,
        smooth: boolean,
    ) => unknown;
    getPosition?: (target: Vector3) => Vector3;
    getTarget?: (target: Vector3) => Vector3;
};

function nearestNode(
    model: ResortModel,
    nodes: typeof model.resort.nodes,
    elevation?: number,
) {
    return [...nodes].sort((left, right) => {
        const leftPosition = model.nodePosition(left);
        const rightPosition = model.nodePosition(right);
        const leftScore =
            (leftPosition.x - model.centre.x) ** 2 +
            (leftPosition.z - model.centre.z) ** 2 +
            (elevation === undefined ? 0 : (left.elevation - elevation) ** 2);
        const rightScore =
            (rightPosition.x - model.centre.x) ** 2 +
            (rightPosition.z - model.centre.z) ** 2 +
            (elevation === undefined ? 0 : (right.elevation - elevation) ** 2);
        return leftScore - rightScore || left.node_id.localeCompare(right.node_id);
    })[0];
}

function poseAround(
    model: ResortModel,
    target: [number, number, number],
    distance: number,
): CameraPose {
    return {
        target,
        position: [
            target[0] + model.planExtent * distance,
            target[1] + model.planExtent * distance * 0.8,
            target[2] - model.planExtent * distance,
        ],
    };
}

export function cameraPresets(model: ResortModel): CameraPresetMap {
    const elevations = model.resort.nodes
        .map((node) => node.elevation)
        .sort((left, right) => left - right);
    const medianElevation = elevations[Math.floor(elevations.length / 2)];
    const zoneNode = nearestNode(model, model.resort.nodes, medianElevation);
    const stations = model.resort.nodes.filter(
        (node) => node.node_type === "lift_station",
    );
    const operationsNode = nearestNode(
        model,
        stations.length > 0 ? stations : model.resort.nodes,
    );
    const zone = model.nodePosition(zoneNode);
    const operations = model.nodePosition(operationsNode);
    return {
        overview: {
            position: model.cameraPosition,
            target: model.cameraTarget,
        },
        zone: poseAround(model, [zone.x, zone.y, zone.z], 0.25),
        operations: poseAround(
            model,
            [operations.x, operations.y, operations.z],
            0.14,
        ),
    };
}

export function moveToPreset(
    controls: CameraControl,
    pose: CameraPose,
    reduceMotion: boolean,
) {
    controls.setLookAt(
        ...pose.position,
        ...pose.target,
        !reduceMotion,
    );
}

function focusTarget(
    request: Exclude<FocusRequest, null>,
    model: ResortModel,
): Vector3 | null {
    const selection = request.selection;
    if (selection.kind === "node" || selection.kind === "building") {
        const node = model.resort.nodes[selection.index];
        return node ? model.nodePosition(node) : null;
    }
    if (selection.kind === "piste" || selection.kind === "lift") {
        const edge = model.resort.edges[selection.index];
        if (!edge) return null;
        const curve = edge.edge_type === "lift"
            ? liftShape(edge, model).cable
            : pisteCurve(edge, model);
        return curve.getPoint(0.5);
    }
    return null;
}

export function focusInfrastructure(
    controls: CameraControl,
    request: Exclude<FocusRequest, null>,
    model: ResortModel,
    reduceMotion: boolean,
): boolean {
    const target = focusTarget(request, model);
    if (!target) return false;
    const currentPosition = controls.getPosition?.(new Vector3());
    const currentTarget = controls.getTarget?.(new Vector3());
    const position = currentPosition && currentTarget
        ? target.clone().add(currentPosition.clone().sub(currentTarget))
        : new Vector3(...poseAround(model, [target.x, target.y, target.z], 0.18).position);
    controls.setLookAt(
        position.x,
        position.y,
        position.z,
        target.x,
        target.y,
        target.z,
        !reduceMotion,
    );
    return true;
}
