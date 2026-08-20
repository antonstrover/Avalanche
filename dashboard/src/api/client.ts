import type { operations } from "./schema";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

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
