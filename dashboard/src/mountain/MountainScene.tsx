import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
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
import { selectionLabel, type FocusRequest, type Selection } from "./selection";
import type { LiveSession } from "../api/client";
import type { DisplayState } from "../workers/live-frame";
import {
    INITIAL_LAYERS,
    LAYER_NAMES,
    type LayerName,
    type LayerVisibility,
} from "./layers";
import { divergentEdges, edgeTelemetryView } from "./telemetryView";
import { DivergenceMarkers } from "./Divergence";
import { InterventionHighlights, RouteOverlay } from "./RouteOverlay";
import {
    cameraPresets,
    focusInfrastructure,
    moveToPreset,
    type CameraPresetName,
} from "./cameraPresets";
import { reducedMotion } from "./conditions";
import {
    OrbitCameraControls,
    type OrbitCameraHandle,
} from "./OrbitCameraControls";

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
    selection,
    onSelectionChange,
    focusRequest = null,
    onDecisionFocus = () => undefined,
}: {
    session: LiveSession | null;
    display: DisplayState;
    onLiveFrame: (count: number, display: DisplayState) => void;
    onLiveError: () => void;
    model?: ResortModel;
    showTrueState: boolean;
    selection: Selection;
    onSelectionChange: (selection: Selection) => void;
    focusRequest?: FocusRequest;
    onDecisionFocus?: () => void;
}) {
    const [drawn, setDrawn] = useState(false);
    const [visibleLayers, setVisibleLayers] = useState(INITIAL_LAYERS);
    const controls = useRef<OrbitCameraHandle>(null);
    const presets = useMemo(() => cameraPresets(model), [model]);
    const telemetry = useMemo(
        () => edgeTelemetryView(display.telemetry, showTrueState),
        [display.telemetry, showTrueState],
    );
    const divergent = useMemo(
        () => divergentEdges(display.telemetry, display.attack),
        [display.telemetry, display.attack],
    );

    useEffect(() => {
        if (!focusRequest || !controls.current) return;
        try {
            focusInfrastructure(
                controls.current,
                focusRequest,
                model,
                reducedMotion(),
            );
        } catch {
            // Keep the selected item when camera movement fails.
        }
    }, [focusRequest, model]);

    return (
        <section className="mountain">
            <div
                className="mountain-canvas"
                data-testid="mountain-canvas"
                data-drawn={drawn}
                data-state-view={showTrueState ? "true" : "reported"}
                data-attack-kind={display.attack.kind}
                data-attack-active={display.attack.active}
                data-divergent-edges={divergent.length}
            >
                <Canvas
                    camera={{ position: model.cameraPosition, fov: 45, near: 1, far: 1000 }}
                    onPointerMissed={() => onSelectionChange(null)}
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
                            onSelect={onSelectionChange}
                            model={model}
                        />
                        <Lifts
                            closedEdges={telemetry.closedEdges}
                            selection={selection}
                            onSelect={onSelectionChange}
                            model={model}
                        />
                    </group>
                    <group
                        name="infrastructure-layer"
                        visible={visibleLayers.infrastructure}
                    >
                        <Buildings
                            selection={selection}
                            onSelect={onSelectionChange}
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
                        <DivergenceMarkers
                            telemetry={display.telemetry}
                            attack={display.attack}
                            model={model}
                        />
                        <Hazards
                            hazards={display.hazards}
                            selection={selection}
                            onSelect={onSelectionChange}
                            model={model}
                        />
                        <Failures failures={display.failures} model={model} />
                    </group>
                    <group
                        name="recommendations-layer"
                        visible={visibleLayers.recommendations}
                    >
                        <RouteOverlay decision={display.decision} model={model} />
                    </group>
                    <group name="selection-layer" visible={visibleLayers.selection}>
                        <InterventionHighlights
                            decision={display.decision}
                            model={model}
                            onSelect={onSelectionChange}
                            onFocus={onDecisionFocus}
                        />
                    </group>
                    <OrbitCameraControls
                        ref={controls}
                        initialPosition={model.cameraPosition}
                        initialTarget={model.cameraTarget}
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
                {display.decision && (
                    <div className="route-legend" aria-label="Route overlay legend">
                        <span><i className="proposed-route" />Proposed route</span>
                        <span><i className="executed-route" />Executed route</span>
                    </div>
                )}
                {divergent.length > 0 && (
                    <div
                        className="divergence-legend"
                        data-testid="divergence-legend"
                        aria-label="Telemetry divergence legend"
                    >
                        <span>
                            <i className="divergence-marker" />
                            Ring marker: the report and the truth differ
                        </span>
                    </div>
                )}
            </div>
            <Timeline events={display.timeline} weather={display.weather} />
        </section>
    );
}
