import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { BufferAttribute, BufferGeometry, type Points } from "three";
import { reducedMotion } from "./conditions";
import { centre } from "./resort";
import type { WeatherState } from "../workers/live-frame";

const FLAKE_COUNT = 1400;
const FIELD_SIZE = 240;
const FIELD_HEIGHT = 110;
const FOG_COLOUR = "#c7d8ea";

// The flake positions come from a fixed sequence, so each run looks the same.
function pseudoRandom(index: number): number {
    const value = Math.sin(index * 12.9898) * 43758.5453;
    return value - Math.floor(value);
}

export function Weather({ weather }: { weather: WeatherState }) {
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
            let height = position.getY(index) - delta * (2 + 3 * weather.snowfall);
            let across = position.getX(index) + delta * weather.wind * 0.4;
            if (height < 0) height += FIELD_HEIGHT;
            if (across > centre.x + FIELD_SIZE / 2) across -= FIELD_SIZE;
            position.setY(index, height);
            position.setX(index, across);
        }
        position.needsUpdate = true;
    });

    // A low visibility brings the fog near to the camera.
    const fogNear = Math.max(12, Math.min(weather.visibility * 0.04, 180));
    const fogFar = Math.max(fogNear + 20, Math.min(weather.visibility * 0.16, 500));
    const snow = Math.min(weather.snowfall, 6);
    return (
        <>
            <fog
                attach="fog"
                args={[FOG_COLOUR, fogNear, fogFar]}
            />
            <points ref={points} geometry={geometry} name="snowfall">
                <pointsMaterial
                    color="#ffffff"
                    size={0.5 + snow * 0.25}
                    sizeAttenuation
                    transparent
                    opacity={snow > 0 ? Math.min(0.35 + snow * 0.1, 0.9) : 0}
                />
            </points>
        </>
    );
}
