import { useCallback, useEffect, useState } from "react";
import {
    createLiveSession,
    fetchHealth,
    type HealthResponse,
    type LiveSession,
} from "./api/client";
import { MountainScene } from "./mountain/MountainScene";
import { INITIAL_DISPLAY } from "./mountain/conditions";
import { resort, resortName } from "./mountain/resort";
import { mergeTimeline } from "./features/timeline";
import { DecisionInspector } from "./features/live/DecisionInspector";
import type { DisplayState } from "./workers/live-frame";

function App() {
    const [health, setHealth] = useState<HealthResponse | null>(null);
    const [session, setSession] = useState<LiveSession | null>(null);
    const [liveStatus, setLiveStatus] = useState("idle");
    const [liveCount, setLiveCount] = useState(0);
    const [display, setDisplay] = useState<DisplayState>(INITIAL_DISPLAY);

    useEffect(() => {
        fetchHealth().then(setHealth);
    }, []);

    const startSession = async (demoFailure = false, demoMonitor = false) => {
        setLiveStatus("starting");
        setDisplay(INITIAL_DISPLAY);
        try {
            const created = await createLiveSession(0, 5000, demoFailure, demoMonitor);
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

    return (
        <main>
            <h1>Avalanche control centre</h1>
            <p data-testid="resort-name">
                {resortName} · {resort.nodes.length} nodes · {resort.edges.length} edges
            </p>
            <p data-testid="health-status">API status: {health?.status ?? "loading"}</p>
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
                    onClick={() => startSession(true)}
                    disabled={liveStatus !== "idle"}
                >
                    Start failure demo
                </button>
                <p data-testid="live-status">Live status: {liveStatus}</p>
                <p data-testid="live-skier-count">Live skiers: {liveCount}</p>
            </div>
            <MountainScene
                session={session}
                display={display}
                onLiveFrame={onLiveFrame}
                onLiveError={onLiveError}
            />
            <DecisionInspector decision={display.decision} telemetry={display.telemetry} />
        </main>
    );
}

export default App;
