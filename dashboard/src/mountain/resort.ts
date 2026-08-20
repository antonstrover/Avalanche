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

// The elevation of the base of the mountain, and the vertical scale.
// The plan scale brings the resort metres into the size of the scene.
const BASE_ELEVATION = 940;
const HEIGHT_SCALE = 0.045;
const PLAN_SCALE = 0.09;

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

// The centre of the resort in the plan, used by the camera and the terrain.
export const centre = new Vector3(
    resort.nodes.reduce((sum, node) => sum + nodePosition(node).x, 0) / resort.nodes.length,
    0,
    resort.nodes.reduce((sum, node) => sum + nodePosition(node).z, 0) / resort.nodes.length,
);
