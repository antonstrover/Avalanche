import { useMemo, useRef, useState, type ComponentRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { Buildings } from "./Buildings";
import { Timeline } from "../components/Timeline";
import { Failures } from "./Failures";
import { Hazards } from "./Hazards";
import { Lifts } from "./Lifts";
import { Pistes } from "./Pistes";
import { Skiers } from "./Skiers";
import { Terrain } from "./Terrain";
import { Weather } from "./Weather";
import { cameraPosition, cameraTarget } from "./resort";
import { selectionLabel, type Selection } from "./selection";
import type { LiveSession } from "../api/client";
import type { DisplayState } from "../workers/live-frame";

// The scene reports the first drawn frame. A browser test waits for it.
function FirstFrame({ onDrawn }: { onDrawn: () => void }) {
    const done = useRef(false);
    useFrame(() => {
        if (done.current) return;
        done.current = true;
        onDrawn();
    });
    return null;
}

export function MountainScene({
    session,
    display,
    onLiveFrame,
    onLiveError,
}: {
    session: LiveSession | null;
    display: DisplayState;
    onLiveFrame: (count: number, display: DisplayState) => void;
    onLiveError: () => void;
}) {
    const [selection, setSelection] = useState<Selection>(null);
    const [drawn, setDrawn] = useState(false);
    const controls = useRef<ComponentRef<typeof OrbitControls>>(null);
    const closedEdges = useMemo(
        () => new Set(display.closures.map((closure) => closure.edge_index)),
        [display.closures],
    );

    return (
        <section className="mountain">
            <div className="mountain-canvas" data-testid="mountain-canvas" data-drawn={drawn}>
                <Canvas
                    camera={{ position: cameraPosition, fov: 45, near: 1, far: 1000 }}
                    onPointerMissed={() => setSelection(null)}
                >
                    <color attach="background" args={["#9fc4e8"]} />
                    <hemisphereLight args={["#dfeaff", "#5b5347", 1.1]} />
                    <directionalLight position={[120, 160, 80]} intensity={1.6} />
                    <Terrain />
                    <Pistes
                        closedEdges={closedEdges}
                        selection={selection}
                        onSelect={setSelection}
                    />
                    <Lifts
                        closedEdges={closedEdges}
                        selection={selection}
                        onSelect={setSelection}
                    />
                    <Buildings selection={selection} onSelect={setSelection} />
                    <Hazards
                        hazards={display.hazards}
                        selection={selection}
                        onSelect={setSelection}
                    />
                    <Failures failures={display.failures} />
                    <Skiers
                        session={session}
                        onLiveFrame={onLiveFrame}
                        onLiveError={onLiveError}
                    />
                    <Weather weather={display.weather} />
                    <OrbitControls
                        ref={controls}
                        target={cameraTarget}
                        enablePan
                        enableRotate
                        enableZoom
                        enableDamping={false}
                    />
                    <FirstFrame onDrawn={() => setDrawn(true)} />
                </Canvas>
            </div>
            <div className="mountain-bar">
                <p data-testid="selection">{selectionLabel(selection)}</p>
                <button type="button" onClick={() => controls.current?.reset()}>
                    Reset the view
                </button>
            </div>
            <Timeline events={display.timeline} weather={display.weather} />
        </section>
    );
}
