import type { DisplayState } from "../workers/live-frame";

export const INITIAL_DISPLAY: DisplayState = {
    weather: { wind: 0, visibility: 10_000, snowfall: 0, temperature: 5 },
    failures: [],
    hazards: [],
    closures: [],
    timeline: [],
};

// A person can ask the system to reduce the motion.
// The snow and the skier markers then hold still.
export function reducedMotion(): boolean {
    return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
}
