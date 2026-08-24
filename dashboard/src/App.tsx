import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
    commandLiveSession,
    createLiveSession,
    fetchConfigOptions,
    fetchHealth,
    resolveLiveConfig,
    type ConfigOptionsResponse,
    type HealthResponse,
    type LiveSession,
    type LiveConfigSelection,
    type ResolvedLiveConfig,
} from "./api/client";
import { MountainScene } from "./mountain/MountainScene";
import { INITIAL_DISPLAY } from "./mountain/conditions";
import {
    createResortModel,
    defaultResortModel,
    type Resort,
} from "./mountain/resort";
import { mergeTimeline } from "./features/timeline";
import { DecisionInspector } from "./features/live/DecisionInspector";
import { ApprovalPanel } from "./features/live/ApprovalPanel";
import { SessionSetup } from "./features/live/SessionSetup";
import type { DisplayState } from "./workers/live-frame";

function App() {
    const [health, setHealth] = useState<HealthResponse | null>(null);
    const [configOptions, setConfigOptions] = useState<ConfigOptionsResponse | null>(null);
    const [configFailed, setConfigFailed] = useState(false);
    const [selection, setSelection] = useState<LiveConfigSelection>({
        mountain: "medium-resort",
        scenario: "default",
        controller: "honest",
        monitor: "none",
        seed: 0,
        skier_count: 5000,
    });
    const [resolvedConfig, setResolvedConfig] = useState<ResolvedLiveConfig | null>(null);
    const [session, setSession] = useState<LiveSession | null>(null);
    const [liveStatus, setLiveStatus] = useState("idle");
    const [liveCount, setLiveCount] = useState(0);
    const [simulationSpeed, setSimulationSpeed] = useState(20);
    const [display, setDisplay] = useState<DisplayState>(INITIAL_DISPLAY);
    const [showTrueState, setShowTrueState] = useState(false);
    const decisionInspector = useRef<HTMLDivElement>(null);
    const selectedMountain = configOptions?.mountains.find(
        (option) => option.id === selection.mountain,
    );
    const resortModel = useMemo(
        () =>
            selectedMountain
                ? createResortModel(selectedMountain.topology as unknown as Resort)
                : defaultResortModel,
        [selectedMountain],
    );

    useEffect(() => {
        fetchHealth().then(setHealth);
        fetchConfigOptions().then(setConfigOptions).catch(() => setConfigFailed(true));
    }, []);

    useEffect(() => {
        if (!configOptions) return;
        let current = true;
        resolveLiveConfig(selection)
            .then((resolved) => {
                if (current) setResolvedConfig(resolved);
            })
            .catch(() => {
                if (current) setConfigFailed(true);
            });
        return () => {
            current = false;
        };
    }, [configOptions, selection]);

    const startSession = async (
        demoFailure = false,
        demoMonitor = false,
        demoApproval = false,
    ) => {
        if (!resolvedConfig) return;
        setLiveStatus("starting");
        setDisplay(INITIAL_DISPLAY);
        try {
            const created = await createLiveSession(
                resolvedConfig,
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
        setLiveStatus((current) => (current === "paused" ? current : "live"));
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
                {resortModel.resortName} · {resortModel.resort.nodes.length} nodes ·{" "}
                {resortModel.resort.edges.length} edges
            </p>
            <p data-testid="health-status">API status: {health?.status ?? "loading"}</p>
            <SessionSetup
                options={configOptions}
                selection={selection}
                resolved={resolvedConfig}
                failed={configFailed}
                onChange={(next) => {
                    setResolvedConfig(null);
                    setSelection(next);
                }}
            />
            <div className="live-controls">
                <button
                    type="button"
                    onClick={() => startSession(false)}
                    disabled={liveStatus !== "idle" || !resolvedConfig}
                >
                    Start live session
                </button>
                <button
                    type="button"
                    onClick={() => startSession(false, true)}
                    disabled={liveStatus !== "idle" || !resolvedConfig}
                >
                    Start monitor demo
                </button>
                <button
                    type="button"
                    onClick={() => startSession(false, false, true)}
                    disabled={liveStatus !== "idle" || !resolvedConfig}
                >
                    Start approval demo
                </button>
                <button
                    type="button"
                    onClick={() => startSession(true)}
                    disabled={liveStatus !== "idle" || !resolvedConfig}
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
                model={resortModel}
                showTrueState={showTrueState}
                onDecisionFocus={() =>
                    decisionInspector.current?.scrollIntoView({ block: "nearest" })
                }
            />
            <div ref={decisionInspector}>
                <DecisionInspector decision={display.decision} telemetry={display.telemetry} />
            </div>
            <ApprovalPanel decision={display.decision} session={session} />
        </main>
    );
}

export default App;
