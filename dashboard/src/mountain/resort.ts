import { Vector3 } from "three";
import data from "./resort.json";

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

export type ResortModel = {
    resort: Resort;
    resortName: string;
    planExtent: number;
    terrainSize: number;
    centre: Vector3;
    meanNodeSpacing: number;
    cameraTarget: [number, number, number];
    cameraPosition: [number, number, number];
    pistes: { edge: Edge; index: number }[];
    lifts: { edge: Edge; index: number }[];
    buildingNodes: { node: Node; index: number }[];
    nodePosition: (node: Node) => Vector3;
    edgeNodes: (edge: Edge) => [Node, Node];
};

const TARGET_SCENE_WIDTH = 240;
const HEIGHT_SCALE = 0.045;
const TERRAIN_MARGIN = 30;

export function createResortModel(resort: Resort): ResortModel {
    const bounds = resort.nodes.reduce(
        (current, node) => ({
            minX: Math.min(current.minX, node.x),
            maxX: Math.max(current.maxX, node.x),
            minY: Math.min(current.minY, node.y),
            maxY: Math.max(current.maxY, node.y),
            minElevation: Math.min(current.minElevation, node.elevation),
            maxElevation: Math.max(current.maxElevation, node.elevation),
        }),
        {
            minX: Number.POSITIVE_INFINITY,
            maxX: Number.NEGATIVE_INFINITY,
            minY: Number.POSITIVE_INFINITY,
            maxY: Number.NEGATIVE_INFINITY,
            minElevation: Number.POSITIVE_INFINITY,
            maxElevation: Number.NEGATIVE_INFINITY,
        },
    );
    const sourceExtent = Math.max(
        bounds.maxX - bounds.minX,
        bounds.maxY - bounds.minY,
    );
    const planScale = TARGET_SCENE_WIDTH / sourceExtent;
    const planExtent = sourceExtent * planScale;
    const nodePosition = (node: Node) =>
        new Vector3(
            node.x * planScale,
            (node.elevation - bounds.minElevation) * HEIGHT_SCALE,
            node.y * planScale,
        );
    const byId = new Map(resort.nodes.map((node) => [node.node_id, node]));
    const edgeNodes = (edge: Edge): [Node, Node] => {
        const source = byId.get(edge.source);
        const destination = byId.get(edge.destination);
        if (!source || !destination) {
            throw new Error(`The edge ${edge.source} to ${edge.destination} is invalid.`);
        }
        return [source, destination];
    };
    const centre = new Vector3(
        ((bounds.minX + bounds.maxX) / 2) * planScale,
        0,
        ((bounds.minY + bounds.maxY) / 2) * planScale,
    );
    const meanNodeSpacing =
        resort.nodes.reduce((sum, node, index) => {
            const position = nodePosition(node);
            const nearest = resort.nodes.reduce((distance, other, otherIndex) => {
                if (index === otherIndex) return distance;
                const otherPosition = nodePosition(other);
                return Math.min(
                    distance,
                    Math.hypot(
                        position.x - otherPosition.x,
                        position.z - otherPosition.z,
                    ),
                );
            }, Number.POSITIVE_INFINITY);
            return sum + nearest;
        }, 0) / resort.nodes.length;
    const sceneHeight = (bounds.maxElevation - bounds.minElevation) * HEIGHT_SCALE;
    const cameraTarget: [number, number, number] = [
        centre.x,
        sceneHeight * 0.25,
        centre.z,
    ];
    const cameraPosition: [number, number, number] = [
        centre.x + planExtent * 0.44,
        sceneHeight * 0.25 + planExtent * 0.36,
        centre.z - planExtent * 0.55,
    ];
    return {
        resort,
        resortName: resort.name.replace(/\b\w/g, (letter) => letter.toUpperCase()),
        planExtent,
        terrainSize: planExtent + 2 * TERRAIN_MARGIN,
        centre,
        meanNodeSpacing,
        cameraTarget,
        cameraPosition,
        pistes: resort.edges
            .map((edge, index) => ({ edge, index }))
            .filter((item) => item.edge.edge_type === "piste"),
        lifts: resort.edges
            .map((edge, index) => ({ edge, index }))
            .filter((item) => item.edge.edge_type === "lift"),
        buildingNodes: resort.nodes
            .map((node, index) => ({ node, index }))
            .filter((item) =>
                ["entrance", "exit", "lift_station"].includes(item.node.node_type),
            ),
        nodePosition,
        edgeNodes,
    };
}

export const defaultResortModel = createResortModel(data as Resort);
export const resort = defaultResortModel.resort;
export const resortName = defaultResortModel.resortName;
export const planExtent = defaultResortModel.planExtent;
export const terrainSize = defaultResortModel.terrainSize;
export const centre = defaultResortModel.centre;
export const meanNodeSpacing = defaultResortModel.meanNodeSpacing;
export const cameraTarget = defaultResortModel.cameraTarget;
export const cameraPosition = defaultResortModel.cameraPosition;
export const pistes = defaultResortModel.pistes;
export const lifts = defaultResortModel.lifts;
export const buildingNodes = defaultResortModel.buildingNodes;
export const nodePosition = defaultResortModel.nodePosition;
export const edgeNodes = defaultResortModel.edgeNodes;

export const difficultyColour: Record<string, string> = {
    green: "#3fae55",
    blue: "#2f7de1",
    red: "#c9452f",
    black: "#22242c",
    none: "#8a8f9a",
};
