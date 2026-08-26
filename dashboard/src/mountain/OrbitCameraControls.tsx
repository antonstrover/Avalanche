import {
    forwardRef,
    useCallback,
    useEffect,
    useImperativeHandle,
    useMemo,
    useRef,
} from "react";
import { useFrame, useThree } from "@react-three/fiber";
import { Vector3 } from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { CameraControl } from "./cameraPresets";
import { advanceCameraPose, type CameraGoal } from "./cameraMotion";

export type OrbitCameraHandle = CameraControl & {
    reset: (smooth: boolean) => void;
};

export const OrbitCameraControls = forwardRef<OrbitCameraHandle, {
    initialPosition: [number, number, number];
    initialTarget: [number, number, number];
}>(function OrbitCameraControls(
    { initialPosition, initialTarget },
    forwardedRef,
) {
    const { camera, gl } = useThree();
    const controls = useMemo(
        () => {
            const value = new OrbitControls(camera, gl.domElement);
            value.enableDamping = true;
            value.dampingFactor = 0.08;
            return value;
        },
        [camera, gl.domElement],
    );
    const goal = useRef<CameraGoal | null>(null);

    useEffect(() => {
        const cancelMotion = () => {
            goal.current = null;
        };
        controls.addEventListener("start", cancelMotion);
        return () => {
            controls.removeEventListener("start", cancelMotion);
            controls.dispose();
        };
    }, [controls]);

    useEffect(() => {
        goal.current = null;
        camera.position.set(...initialPosition);
        controls.target.set(...initialTarget);
        controls.update();
    }, [camera, controls, initialPosition, initialTarget]);

    const setPose = useCallback(
        (position: Vector3, target: Vector3, smooth: boolean) => {
            if (smooth) {
                goal.current = {
                    position: position.clone(),
                    target: target.clone(),
                };
                return;
            }
            goal.current = null;
            camera.position.copy(position);
            controls.target.copy(target);
            controls.update();
        },
        [camera, controls],
    );

    useImperativeHandle(
        forwardedRef,
        () => ({
            setLookAt: (x, y, z, targetX, targetY, targetZ, smooth) => {
                setPose(
                    new Vector3(x, y, z),
                    new Vector3(targetX, targetY, targetZ),
                    smooth,
                );
            },
            getPosition: (target) => target.copy(camera.position),
            getTarget: (target) => target.copy(controls.target),
            reset: (smooth) => {
                setPose(
                    new Vector3(...initialPosition),
                    new Vector3(...initialTarget),
                    smooth,
                );
            },
        }),
        [camera, controls, initialPosition, initialTarget, setPose],
    );

    useFrame(() => {
        if (goal.current) {
            const complete = advanceCameraPose(
                camera.position,
                controls.target,
                goal.current,
            );
            if (complete) goal.current = null;
        }
        controls.update();
    });

    return null;
});
