import { useCallback, useEffect, useState } from "react";
import {
    commandLiveSession,
    createLiveSession,
    fetchConfigOptions,
    fetchHealth,
    type ConfigOptionsResponse,
    type HealthResponse,
    type LiveSession,
} from "./api/client";
import { MountainScene } from "./mountain/MountainScene";
import { INITIAL_DISPLAY } from "./mountain/conditions";
import { resort, resortName } from "./mountain/resort";
import { mergeTimeline } from "./features/timeline";
import { DecisionInspector } from "./features/live/DecisionInspector";
import { ApprovalPanel } from "./features/live/ApprovalPanel";
import { SessionSetup } from "./features/live/SessionSetup";
import type { DisplayState } from "./workers/live-frame";

function App() {
    const [health, setHealth] = useState<HealthResponse | null>(null);
    const [configOptions, setConfigOptions] = useState<ConfigOptionsResponse | null>(null);
    const [configFailed, setConfigFailed] = useState(false);
    const [session, setSession] = useState<LiveSession | null>(null);
    const [liveStatus, setLiveStatus] = useState("idle");
    const [liveCount, setLiveCount] = useState(0);
    const [simulationSpeed, setSimulationSpeed] = useState(20);
    const [display, setDisplay] = useState<DisplayState>(INITIAL_DISPLAY);
    const [showTrueState, setShowTrueState] = useState(false);

    useEffect(() => {
        fetchHealth().then(setHealth);
        fetchConfigOptions().then(setConfigOptions).catch(() => setConfigFailed(true));
    }, []);

    const startSession = async (
        demoFailure = false,
        demoMonitor = false,
        demoApproval = false,
    ) => {
        setLiveStatus("starting");
        setDisplay(INITIAL_DISPLAY);
        try {
            const created = await createLiveSession(
                0,
                5000,
                demoFailure,
                demoMonitor,
                demoApproval,
            );
            setSession(created);
            setLiveStatus("connecting");
        } catch {
            setLiveStatus("failed");
        }
    };

    const onLiveFrame = useCallback((count: number, next: DisplayState) => {
        setLiveCount(count);
        setLiveStatus("live");
        setDisplay((current) => ({
            ...next,
            timeline: mergeTimeline(current.timeline, next.timeline),
        }));
    }, []);
    const onLiveError = useCallback(() => setLiveStatus("failed"), []);

    const sendCommand = async (
        command: "pause" | "resume" | "step" | "set_speed",
    ) => {
        if (!session) return;
        try {
            const updated = await commandLiveSession(
                session.session_id,
                command,
                command === "set_speed" ? simulationSpeed : undefined,
            );
            setSession(updated);
            setLiveStatus(updated.status);
        } catch {
            setLiveStatus("failed");
        }
    };

    return (
        <main>
            <h1>Avalanche control centre</h1>
            <p data-testid="resort-name">
                {resortName} · {resort.nodes.length} nodes · {resort.edges.length} edges
            </p>
            <p data-testid="health-status">API status: {health?.status ?? "loading"}</p>
            <SessionSetup options={configOptions} failed={configFailed} />
            <div className="live-controls">
                <button
                    type="button"
                    onClick={() => startSession(false)}
                    disabled={liveStatus !== "idle"}
                >
                    Start live session
                </button>
                <button
                    type="button"
                    onClick={() => startSession(false, true)}
                    disabled={liveStatus !== "idle"}
                >
                    Start monitor demo
                </button>
                <button
                    type="button"
                    onClick={() => startSession(false, false, true)}
                    disabled={liveStatus !== "idle"}
                >
                    Start approval demo
                </button>
                <button
                    type="button"
                    onClick={() => startSession(true)}
                    disabled={liveStatus !== "idle"}
                >
                    Start failure demo
                </button>
                <button
                    type="button"
                    onClick={() => sendCommand("pause")}
                    disabled={!session || !["live", "running"].includes(liveStatus)}
                >
                    Pause
                </button>
                <button
                    type="button"
                    onClick={() => sendCommand("resume")}
                    disabled={!session || liveStatus !== "paused"}
                >
                    Resume
                </button>
                <button
                    type="button"
                    onClick={() => sendCommand("step")}
                    disabled={!session || liveStatus !== "paused"}
                >
                    Step
                </button>
                <label>
                    Simulation speed
                    <input
                        type="number"
                        min="0.1"
                        step="0.1"
                        value={simulationSpeed}
                        onChange={(event) => setSimulationSpeed(Number(event.target.value))}
                    />
                </label>
                <button
                    type="button"
                    onClick={() => sendCommand("set_speed")}
                    disabled={!session || !Number.isFinite(simulationSpeed) || simulationSpeed <= 0}
                >
                    Set speed
                </button>
                <p data-testid="live-status">Live status: {liveStatus}</p>
                <p data-testid="live-skier-count">Live skiers: {liveCount}</p>
                <label>
                    <input
                        type="checkbox"
                        checked={showTrueState}
                        onChange={(event) => setShowTrueState(event.target.checked)}
                    />
                    Show the true state
                </label>
            </div>
            <MountainScene
                session={session}
                display={display}
                onLiveFrame={onLiveFrame}
                onLiveError={onLiveError}
                showTrueState={showTrueState}
            />
            <DecisionInspector decision={display.decision} telemetry={display.telemetry} />
            <ApprovalPanel decision={display.decision} session={session} />
        </main>
    );
}

export default App;
