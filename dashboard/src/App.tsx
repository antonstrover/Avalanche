import { useEffect, useState } from "react";
import { fetchHealth, type HealthResponse } from "./api/client";

function App() {
    const [health, setHealth] = useState<HealthResponse | null>(null);

    useEffect(() => {
        fetchHealth().then(setHealth);
    }, []);

    return (
        <main>
            <h1>Avalanche control centre</h1>
            <p data-testid="health-status">API status: {health?.status ?? "loading"}</p>
        </main>
    );
}

export default App;
