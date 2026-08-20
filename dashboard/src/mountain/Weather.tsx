import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { BufferAttribute, BufferGeometry, type Points } from "three";
import { weather } from "./conditions";
import { centre } from "./resort";

const FLAKE_COUNT = 1400;
const FIELD_SIZE = 240;
const FIELD_HEIGHT = 110;
const FOG_COLOUR = "#c7d8ea";

// The flake positions come from a fixed sequence, so each run looks the same.
function pseudoRandom(index: number): number {
    const value = Math.sin(index * 12.9898) * 43758.5453;
    return value - Math.floor(value);
}

// A person can ask the system to reduce the motion. The snow then holds still.
function reducedMotion(): boolean {
    return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
}

export function Weather() {
    const points = useRef<Points>(null);

    const geometry = useMemo(() => {
        const positions = new Float32Array(FLAKE_COUNT * 3);
        for (let index = 0; index < FLAKE_COUNT; index += 1) {
            positions[index * 3] = centre.x + (pseudoRandom(index) - 0.5) * FIELD_SIZE;
            positions[index * 3 + 1] = pseudoRandom(index + FLAKE_COUNT) * FIELD_HEIGHT;
            positions[index * 3 + 2] =
                centre.z + (pseudoRandom(index + 2 * FLAKE_COUNT) - 0.5) * FIELD_SIZE;
        }
        const buffer = new BufferGeometry();
        buffer.setAttribute("position", new BufferAttribute(positions, 3));
        return buffer;
    }, []);

    const still = useMemo(() => reducedMotion(), []);

    // The snowfall sets the fall speed. The wind pushes the flakes sideways.
    useFrame((_state, delta) => {
        if (still || !points.current) return;
        const position = points.current.geometry.attributes.position as BufferAttribute;
        for (let index = 0; index < FLAKE_COUNT; index += 1) {
            let height = position.getY(index) - delta * (4 + 12 * weather.snowfall);
            let across = position.getX(index) + delta * weather.wind * 0.4;
            if (height < 0) height += FIELD_HEIGHT;
            if (across > centre.x + FIELD_SIZE / 2) across -= FIELD_SIZE;
            position.setY(index, height);
            position.setX(index, across);
        }
        position.needsUpdate = true;
    });

    // A low visibility brings the fog near to the camera.
    return (
        <>
            <fog
                attach="fog"
                args={[FOG_COLOUR, weather.visibility * 0.25, weather.visibility * 1.6]}
            />
            <points ref={points} geometry={geometry} name="snowfall">
                <pointsMaterial
                    color="#ffffff"
                    size={0.7 + weather.snowfall}
                    sizeAttenuation
                    transparent
                    opacity={0.85}
                />
            </points>
        </>
    );
}
