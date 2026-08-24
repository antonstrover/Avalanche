import type { ResortModel } from "./resort";

export type CameraPresetName = "overview" | "zone" | "operations";

export type CameraPose = {
    position: [number, number, number];
    target: [number, number, number];
};

export type CameraPresetMap = Record<CameraPresetName, CameraPose>;

type CameraControl = {
    setLookAt: (
        x: number,
        y: number,
        z: number,
        targetX: number,
        targetY: number,
        targetZ: number,
        smooth: boolean,
    ) => unknown;
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
