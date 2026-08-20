import { useCallback, useEffect, useState } from "react";
import {
    createLiveSession,
    fetchHealth,
    type HealthResponse,
    type LiveSession,
} from "./api/client";
import { MountainScene } from "./mountain/MountainScene";

function App() {
    const [health, setHealth] = useState<HealthResponse | null>(null);
    const [session, setSession] = useState<LiveSession | null>(null);
    const [liveStatus, setLiveStatus] = useState("idle");
    const [liveCount, setLiveCount] = useState(0);

    useEffect(() => {
        fetchHealth().then(setHealth);
    }, []);

    const startSession = async () => {
        setLiveStatus("starting");
        try {
            const created = await createLiveSession();
            setSession(created);
            setLiveStatus("connecting");
        } catch {
            setLiveStatus("failed");
        }
    };

    const onLiveFrame = useCallback((count: number) => {
        setLiveCount(count);
        setLiveStatus("live");
    }, []);
    const onLiveError = useCallback(() => setLiveStatus("failed"), []);

    return (
        <main>
            <h1>Avalanche control centre</h1>
            <p data-testid="health-status">API status: {health?.status ?? "loading"}</p>
            <div className="live-controls">
                <button type="button" onClick={startSession} disabled={liveStatus !== "idle"}>
                    Start live session
                </button>
                <p data-testid="live-status">Live status: {liveStatus}</p>
                <p data-testid="live-skier-count">Live skiers: {liveCount}</p>
            </div>
            <MountainScene
                session={session}
                onLiveFrame={onLiveFrame}
                onLiveError={onLiveError}
            />
        </main>
    );
}

export default App;
