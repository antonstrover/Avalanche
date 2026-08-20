import { useRef, useState, type ComponentRef } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { Buildings } from "./Buildings";
import { Lifts } from "./Lifts";
import { Pistes } from "./Pistes";
import { Terrain } from "./Terrain";
import { selectionLabel, type Selection } from "./selection";

// The camera preset gives the overview. The reset button returns to it.
const CAMERA_POSITION: [number, number, number] = [152, 100, -106];
const CAMERA_TARGET: [number, number, number] = [46, 14, 25];

export function MountainScene() {
    const [selection, setSelection] = useState<Selection>(null);
    const controls = useRef<ComponentRef<typeof OrbitControls>>(null);

    return (
        <section className="mountain">
            <div className="mountain-canvas" data-testid="mountain-canvas">
                <Canvas
                    camera={{ position: CAMERA_POSITION, fov: 45, near: 1, far: 1000 }}
                    onPointerMissed={() => setSelection(null)}
                >
                    <color attach="background" args={["#9fc4e8"]} />
                    <hemisphereLight args={["#dfeaff", "#5b5347", 1.1]} />
                    <directionalLight position={[120, 160, 80]} intensity={1.6} />
                    <Terrain />
                    <Pistes selection={selection} onSelect={setSelection} />
                    <Lifts selection={selection} onSelect={setSelection} />
                    <Buildings selection={selection} onSelect={setSelection} />
                    <OrbitControls
                        ref={controls}
                        target={CAMERA_TARGET}
                        enablePan
                        enableRotate
                        enableZoom
                        enableDamping={false}
                    />
                </Canvas>
            </div>
            <div className="mountain-bar">
                <p data-testid="selection">{selectionLabel(selection)}</p>
                <button type="button" onClick={() => controls.current?.reset()}>
                    Reset the view
                </button>
            </div>
        </section>
    );
}
