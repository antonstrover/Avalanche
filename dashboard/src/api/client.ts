import type { components, operations } from "./schema";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export type HealthResponse =
    operations["health_health_get"]["responses"][200]["content"]["application/json"];

export type ConfigOptionsResponse =
    operations["config_options_api_config_options_get"]["responses"][200]["content"]["application/json"];

export async function fetchHealth(): Promise<HealthResponse> {
    const response = await fetch(`${API_BASE}/health`);
    return response.json();
}

export async function fetchConfigOptions(): Promise<ConfigOptionsResponse> {
    const response = await fetch(`${API_BASE}/api/config-options`);
    return response.json();
}

export type LiveSession = components["schemas"]["SessionResponse"];

export async function createLiveSession(
    seed = 0,
    skierCount = 5000,
    demoFailure = false,
): Promise<LiveSession> {
    const response = await fetch(`${API_BASE}/api/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ seed, skier_count: skierCount, demo_failure: demoFailure }),
    });
    if (!response.ok) throw new Error("the live session could not start");
    return response.json() as Promise<LiveSession>;
}

export function liveStreamUrl(sessionId: string): string {
    const base = API_BASE || window.location.origin;
    const url = new URL(`/api/sessions/${sessionId}/stream`, base);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    return url.toString();
}
