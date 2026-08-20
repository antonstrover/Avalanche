import { useCallback, useEffect, useState } from "react";
import {
    createLiveSession,
    fetchHealth,
    type HealthResponse,
    type LiveSession,
} from "./api/client";
import { MountainScene } from "./mountain/MountainScene";
import { INITIAL_DISPLAY } from "./mountain/conditions";
import { mergeTimeline } from "./features/timeline";
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

    const startSession = async (demoFailure = false) => {
        setLiveStatus("starting");
        setDisplay(INITIAL_DISPLAY);
        try {
            const created = await createLiveSession(0, 5000, demoFailure);
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
        </main>
    );
}

export default App;
