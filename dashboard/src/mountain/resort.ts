import { Vector3 } from "three";
import data from "./resort.json";

// The scene draws the resort graph. It is not a second physics engine.
// A node keeps its plan position and its elevation from the mountain file.

export type Node = {
    node_id: string;
    x: number;
    y: number;
    elevation: number;
    node_type: string;
    capacity: number;
};

export type Edge = {
    source: string;
    destination: string;
    edge_type: string;
    difficulty: string;
    length: number;
};

export type Resort = { name: string; nodes: Node[]; edges: Edge[] };

export const resort: Resort = data;

// The plan scale fits each resort into the same scene width.
const TARGET_SCENE_WIDTH = 240;
const HEIGHT_SCALE = 0.045;
const TERRAIN_MARGIN = 30;

const planBounds = resort.nodes.reduce(
    (bounds, node) => ({
        minX: Math.min(bounds.minX, node.x),
        maxX: Math.max(bounds.maxX, node.x),
        minY: Math.min(bounds.minY, node.y),
        maxY: Math.max(bounds.maxY, node.y),
        minElevation: Math.min(bounds.minElevation, node.elevation),
    }),
    {
        minX: Number.POSITIVE_INFINITY,
        maxX: Number.NEGATIVE_INFINITY,
        minY: Number.POSITIVE_INFINITY,
        maxY: Number.NEGATIVE_INFINITY,
        minElevation: Number.POSITIVE_INFINITY,
    },
);

const sourcePlanExtent = Math.max(
    planBounds.maxX - planBounds.minX,
    planBounds.maxY - planBounds.minY,
);
const PLAN_SCALE = TARGET_SCENE_WIDTH / sourcePlanExtent;
const BASE_ELEVATION = planBounds.minElevation;

export const planExtent = sourcePlanExtent * PLAN_SCALE;
export const terrainSize = planExtent + 2 * TERRAIN_MARGIN;

export function nodePosition(node: Node): Vector3 {
    return new Vector3(
        node.x * PLAN_SCALE,
        (node.elevation - BASE_ELEVATION) * HEIGHT_SCALE,
        node.y * PLAN_SCALE,
    );
}

const byId = new Map(resort.nodes.map((node) => [node.node_id, node]));

export function edgeNodes(edge: Edge): [Node, Node] {
    const source = byId.get(edge.source);
    const destination = byId.get(edge.destination);
    if (!source || !destination) {
        throw new Error(`The edge ${edge.source} to ${edge.destination} names an unknown node.`);
    }
    return [source, destination];
}

export const pistes = resort.edges
    .map((edge, index) => ({ edge, index }))
    .filter((item) => item.edge.edge_type === "piste");

export const lifts = resort.edges
    .map((edge, index) => ({ edge, index }))
    .filter((item) => item.edge.edge_type === "lift");

// A building stands at an entrance, at an exit, and at each lift station.
export const buildingNodes = resort.nodes
    .map((node, index) => ({ node, index }))
    .filter((item) => ["entrance", "exit", "lift_station"].includes(item.node.node_type));

export const difficultyColour: Record<string, string> = {
    green: "#3fae55",
    blue: "#2f7de1",
    red: "#c9452f",
    black: "#22242c",
    none: "#8a8f9a",
};

// The centre of the bounds gives the terrain and the camera one stable target.
export const centre = new Vector3(
    ((planBounds.minX + planBounds.maxX) / 2) * PLAN_SCALE,
    0,
    ((planBounds.minY + planBounds.maxY) / 2) * PLAN_SCALE,
);

const sceneHeight = (Math.max(...resort.nodes.map((node) => node.elevation)) - BASE_ELEVATION)
    * HEIGHT_SCALE;

export const cameraTarget: [number, number, number] = [
    centre.x,
    sceneHeight * 0.25,
    centre.z,
];

export const cameraPosition: [number, number, number] = [
    centre.x + planExtent * 0.44,
    sceneHeight * 0.25 + planExtent * 0.36,
    centre.z - planExtent * 0.55,
];
