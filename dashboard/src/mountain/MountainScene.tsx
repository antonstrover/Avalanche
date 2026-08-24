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
import { defaultResortModel, type ResortModel } from "./resort";
import { selectionLabel, type Selection } from "./selection";
import type { LiveSession } from "../api/client";
import type { DisplayState } from "../workers/live-frame";
import {
    INITIAL_LAYERS,
    LAYER_NAMES,
    type LayerName,
    type LayerVisibility,
} from "./layers";

export function LayerToggles({
    layers,
    onChange,
}: {
    layers: LayerVisibility;
    onChange: (name: LayerName, visible: boolean) => void;
}) {
    return (
        <fieldset className="layer-toggles">
            <legend>Scene layers</legend>
            {LAYER_NAMES.map((name) => (
                <label key={name}>
                    <input
                        type="checkbox"
                        checked={layers[name]}
                        onChange={(event) => onChange(name, event.target.checked)}
                    />
                    {name}
                </label>
            ))}
        </fieldset>
    );
}

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
    model = defaultResortModel,
}: {
    session: LiveSession | null;
    display: DisplayState;
    onLiveFrame: (count: number, display: DisplayState) => void;
    onLiveError: () => void;
    model?: ResortModel;
}) {
    const [selection, setSelection] = useState<Selection>(null);
    const [drawn, setDrawn] = useState(false);
    const [visibleLayers, setVisibleLayers] = useState(INITIAL_LAYERS);
    const controls = useRef<ComponentRef<typeof OrbitControls>>(null);
    const closedEdges = useMemo(
        () => new Set(display.closures.map((closure) => closure.edge_index)),
        [display.closures],
    );

    return (
        <section className="mountain">
            <div className="mountain-canvas" data-testid="mountain-canvas" data-drawn={drawn}>
                <Canvas
                    camera={{ position: model.cameraPosition, fov: 45, near: 1, far: 1000 }}
                    onPointerMissed={() => setSelection(null)}
                >
                    <color attach="background" args={["#9fc4e8"]} />
                    <hemisphereLight args={["#dfeaff", "#5b5347", 1.1]} />
                    <directionalLight position={[120, 160, 80]} intensity={1.6} />
                    <group name="terrain-layer" visible={visibleLayers.terrain}>
                        <Terrain model={model} />
                    </group>
                    <group name="topology-layer" visible={visibleLayers.topology}>
                        <Pistes
                            closedEdges={closedEdges}
                            selection={selection}
                            onSelect={setSelection}
                            model={model}
                        />
                        <Lifts
                            closedEdges={closedEdges}
                            selection={selection}
                            onSelect={setSelection}
                            model={model}
                        />
                    </group>
                    <group
                        name="infrastructure-layer"
                        visible={visibleLayers.infrastructure}
                    >
                        <Buildings
                            selection={selection}
                            onSelect={setSelection}
                            model={model}
                        />
                    </group>
                    <group name="agents-layer" visible={visibleLayers.agents}>
                        <Skiers
                            session={session}
                            onLiveFrame={onLiveFrame}
                            onLiveError={onLiveError}
                            model={model}
                        />
                    </group>
                    <group name="weather-layer" visible={visibleLayers.weather}>
                        <Weather weather={display.weather} model={model} />
                    </group>
                    <group name="hazards-layer" visible={visibleLayers.hazards}>
                        <Hazards
                            hazards={display.hazards}
                            selection={selection}
                            onSelect={setSelection}
                            model={model}
                        />
                        <Failures failures={display.failures} model={model} />
                    </group>
                    <group
                        name="recommendations-layer"
                        visible={visibleLayers.recommendations}
                    />
                    <group name="selection-layer" visible={visibleLayers.selection} />
                    <OrbitControls
                        ref={controls}
                        target={model.cameraTarget}
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
                <LayerToggles
                    layers={visibleLayers}
                    onChange={(name, visible) =>
                        setVisibleLayers((current) => ({ ...current, [name]: visible }))
                    }
                />
            </div>
            <Timeline events={display.timeline} weather={display.weather} />
        </section>
    );
}
