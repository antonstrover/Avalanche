import { Vector3 } from "three";

const SMOOTH_FRACTION = 0.14;
const COMPLETION_DISTANCE = 0.01;

export type CameraGoal = {
    position: Vector3;
    target: Vector3;
};

export function advanceCameraPose(
    position: Vector3,
    target: Vector3,
    goal: CameraGoal,
): boolean {
    position.lerp(goal.position, SMOOTH_FRACTION);
    target.lerp(goal.target, SMOOTH_FRACTION);
    const complete =
        position.distanceToSquared(goal.position) < COMPLETION_DISTANCE ** 2
        && target.distanceToSquared(goal.target) < COMPLETION_DISTANCE ** 2;
    if (complete) {
        position.copy(goal.position);
        target.copy(goal.target);
    }
    return complete;
}
