import { useMemo, useRef, useState, type ComponentRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { CameraControls } from "@react-three/drei";
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
import { edgeTelemetryView } from "./telemetryView";
import { cameraPresets, moveToPreset, type CameraPresetName } from "./cameraPresets";
import { reducedMotion } from "./conditions";

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
    showTrueState,
}: {
    session: LiveSession | null;
    display: DisplayState;
    onLiveFrame: (count: number, display: DisplayState) => void;
    onLiveError: () => void;
    model?: ResortModel;
    showTrueState: boolean;
}) {
    const [selection, setSelection] = useState<Selection>(null);
    const [drawn, setDrawn] = useState(false);
    const [visibleLayers, setVisibleLayers] = useState(INITIAL_LAYERS);
    const controls = useRef<ComponentRef<typeof CameraControls>>(null);
    const presets = useMemo(() => cameraPresets(model), [model]);
    const telemetry = useMemo(
        () => edgeTelemetryView(display.telemetry, showTrueState),
        [display.telemetry, showTrueState],
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
                            closedEdges={telemetry.closedEdges}
                            density={telemetry.density}
                            selection={selection}
                            onSelect={setSelection}
                            model={model}
                        />
                        <Lifts
                            closedEdges={telemetry.closedEdges}
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
                    <CameraControls
                        ref={controls}
                    />
                    <FirstFrame onDrawn={() => setDrawn(true)} />
                </Canvas>
            </div>
            <div className="mountain-bar">
                <p data-testid="selection">{selectionLabel(selection)}</p>
                <button
                    type="button"
                    onClick={() => controls.current?.reset(!reducedMotion())}
                >
                    Reset the view
                </button>
                {(["overview", "zone", "operations"] as CameraPresetName[]).map(
                    (name) => (
                        <button
                            key={name}
                            type="button"
                            onClick={() => {
                                if (controls.current) {
                                    moveToPreset(
                                        controls.current,
                                        presets[name],
                                        reducedMotion(),
                                    );
                                }
                            }}
                        >
                            {name} view
                        </button>
                    ),
                )}
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
