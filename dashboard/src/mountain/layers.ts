export const LAYER_NAMES = [
    "terrain",
    "topology",
    "infrastructure",
    "agents",
    "weather",
    "hazards",
    "recommendations",
    "selection",
] as const;

export type LayerName = (typeof LAYER_NAMES)[number];
export type LayerVisibility = Record<LayerName, boolean>;

export const INITIAL_LAYERS: LayerVisibility = Object.fromEntries(
    LAYER_NAMES.map((name) => [name, true]),
) as LayerVisibility;
