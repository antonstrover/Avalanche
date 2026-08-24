import type { components, operations } from "./schema";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

export type HealthResponse =
    operations["health_health_get"]["responses"][200]["content"]["application/json"];

export type ConfigOptionsResponse =
    operations["config_options_api_config_options_get"]["responses"][200]["content"]["application/json"];
export type LiveConfigSelection = components["schemas"]["LiveConfigSelection"];
export type ResolvedLiveConfig = components["schemas"]["ResolvedConfig"];

export async function fetchHealth(): Promise<HealthResponse> {
    const response = await fetch(`${API_BASE}/health`);
    return response.json();
}

export async function fetchConfigOptions(): Promise<ConfigOptionsResponse> {
    const response = await fetch(`${API_BASE}/api/config-options`);
    return response.json();
}

export async function resolveLiveConfig(
    selection: LiveConfigSelection,
): Promise<ResolvedLiveConfig> {
    const response = await fetch(`${API_BASE}/api/config-options/resolve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(selection),
    });
    if (!response.ok) throw new Error("the live configuration is invalid");
    return response.json() as Promise<ResolvedLiveConfig>;
}

export type LiveSession = components["schemas"]["SessionResponse"];

export async function createLiveSession(
    config: ResolvedLiveConfig,
    demoFailure = false,
    demoMonitor = false,
    demoApproval = false,
): Promise<LiveSession> {
    const response = await fetch(`${API_BASE}/api/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            config,
            demo_failure: demoFailure,
            demo_monitor: demoMonitor,
            demo_approval: demoApproval,
        }),
    });
    if (!response.ok) throw new Error("the live session could not start");
    return response.json() as Promise<LiveSession>;
}

export type SessionCommand = "pause" | "resume" | "step" | "set_speed";

export async function commandLiveSession(
    sessionId: string,
    command: SessionCommand,
    speed?: number,
): Promise<LiveSession> {
    const response = await fetch(`${API_BASE}/api/sessions/${sessionId}/commands`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ command, speed }),
    });
    if (!response.ok) throw new Error("the live session command failed");
    return response.json() as Promise<LiveSession>;
}

export async function resolveApproval(
    sessionId: string,
    decisionId: string,
    choice: "APPROVE" | "BLOCK" | "REPLACE",
    replacementAction?: unknown,
): Promise<void> {
    const response = await fetch(
        `${API_BASE}/api/sessions/${sessionId}/approvals/${encodeURIComponent(decisionId)}`,
        {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ choice, replacement_action: replacementAction }),
        },
    );
    if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail ?? "the approval response failed");
    }
}

export function liveStreamUrl(sessionId: string): string {
    const base = API_BASE || window.location.origin;
    const url = new URL(`/api/sessions/${sessionId}/stream`, base);
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    return url.toString();
}
