import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { Matrix4, Vector3, type InstancedMesh } from "three";
import { placePosition, type Place } from "./positions";
import { reducedMotion } from "./conditions";
import data from "./replay.sample.json";

// The markers read a recorded run. The scene draws one instanced mesh for them.
// The frame loop writes the positions into the instance buffer.
// A marker never goes through React, so the frame rate stays free of the tick rate.

type Frame = { time: number; skiers: Place[] };
type Replay = { skier_count: number; frames: Frame[] };

const replay = data as unknown as Replay;

const MARKER_RADIUS = 1;
const MARKER_HEIGHT = 3;
// One second of the display holds this many simulated seconds.
const SPEED = 20;
// A person can ask the system to reduce the motion. The markers then hold still.
// A still image test also uses this frame.
const STILL_FRAME = 24;

const HIDDEN = new Matrix4().makeScale(0, 0, 0);

function frameTime(index: number): number {
    return replay.frames[index].time;
}

export function Skiers() {
    const mesh = useRef<InstancedMesh>(null);
    const still = useMemo(() => reducedMotion(), []);
    const time = useRef(frameTime(0));
    const cursor = useRef(0);
    const matrix = useMemo(() => new Matrix4(), []);
    const point = useMemo(() => new Vector3(), []);

    useFrame((_state, delta) => {
        const instances = mesh.current;
        if (!instances) return;

        const first = frameTime(0);
        const last = frameTime(replay.frames.length - 1);
        if (still) {
            time.current = frameTime(STILL_FRAME);
        } else {
            time.current += delta * SPEED;
            if (time.current > last) time.current = first;
        }

        // Step to the frame pair that holds the current time.
        if (frameTime(cursor.current) > time.current) cursor.current = 0;
        while (
            cursor.current + 1 < replay.frames.length &&
            frameTime(cursor.current + 1) <= time.current
        ) {
            cursor.current += 1;
        }

        const before = replay.frames[cursor.current];
        const after = replay.frames[cursor.current + 1] ?? before;
        const span = after.time - before.time;
        const fraction = span > 0 ? (time.current - before.time) / span : 0;

        for (let skier = 0; skier < replay.skier_count; skier += 1) {
            const start = before.skiers[skier];
            const end = after.skiers[skier];
            const startPoint = start ? placePosition(start) : null;
            const endPoint = end ? placePosition(end) : null;

            // Interpolate only along one edge. A change of edge snaps to the new place.
            const sameEdge = start && end && start[0] === end[0] && start[1] === end[1];
            const place =
                startPoint && endPoint && sameEdge
                    ? point.copy(startPoint).lerp(endPoint, fraction)
                    : (endPoint ?? startPoint);

            if (!place) {
                instances.setMatrixAt(skier, HIDDEN);
                continue;
            }
            matrix.makeTranslation(place.x, place.y + MARKER_HEIGHT / 2, place.z);
            instances.setMatrixAt(skier, matrix);
        }
        instances.instanceMatrix.needsUpdate = true;
    });

    return (
        <instancedMesh
            ref={mesh}
            name="skiers"
            args={[undefined, undefined, replay.skier_count]}
            frustumCulled={false}
            raycast={() => null}
        >
            <coneGeometry args={[MARKER_RADIUS, MARKER_HEIGHT, 6]} />
            <meshStandardMaterial color="#1b2233" flatShading roughness={0.6} />
        </instancedMesh>
    );
}
