import { NO_ATTACK, type DisplayState } from "../workers/live-frame";

export const INITIAL_DISPLAY: DisplayState = {
    weather: { wind: 0, visibility: 10_000, snowfall: 0, temperature: 5 },
    failures: [],
    hazards: [],
    closures: [],
    timeline: [],
    decision: null,
    telemetry: {
        reported_density: [],
        true_density: [],
        reported_occupancy: [],
        true_occupancy: [],
        reported_queue: [],
        true_queue: [],
        reported_speed: [],
        true_speed: [],
        reported_closed: [],
        true_closed: [],
    },
    attack: NO_ATTACK,
};

// A person can ask the system to reduce the motion.
// The snow and the skier markers then hold still.
export function reducedMotion(): boolean {
    return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
}
