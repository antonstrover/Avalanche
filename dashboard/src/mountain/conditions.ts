// The hazards and the weather are static placeholders for this stage.
// The simulator does not supply them yet. A later stage connects them.

export type Severity = "low" | "medium" | "high";

// A hazard sits on one edge or on one node of the resort graph.
export type HazardPlace = { kind: "edge"; index: number } | { kind: "node"; node_id: string };

export type Hazard = {
    hazard_id: string;
    place: HazardPlace;
    hazard_kind: string;
    severity: Severity;
};

export const hazards: Hazard[] = [
    {
        hazard_id: "hz-crowding-ridge",
        place: { kind: "edge", index: 6 },
        hazard_kind: "crowding",
        severity: "high",
    },
    {
        hazard_id: "hz-ice-bowl",
        place: { kind: "edge", index: 9 },
        hazard_kind: "ice",
        severity: "medium",
    },
    {
        hazard_id: "hz-wind-summit",
        place: { kind: "node", node_id: "summit_station" },
        hazard_kind: "wind",
        severity: "low",
    },
];

// The wind is in metres each second. The visibility is in metres.
// The snowfall is a value from 0 to 1.
export type Weather = { wind: number; visibility: number; snowfall: number };

export const weather: Weather = { wind: 9, visibility: 260, snowfall: 0.6 };
