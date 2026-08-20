import { useMemo } from "react";
import { BufferAttribute, Color, PlaneGeometry, Vector2, Vector3 } from "three";
import { centre, nodePosition, resort } from "./resort";

const SEGMENTS = 72;
const SIZE = 300;
const CLEARANCE = 1.2;
const SNOW = new Color("#e9f1fb");
const ROCK = new Color("#6f6a67");

// The terrain is a low-poly slope. The nodes give it its shape.
// A fall line gives the general slope. The nodes then pull the surface to them.
function buildSurface(points: Vector3[]) {
    let base = points[0];
    let top = points[0];
    for (const point of points) {
        if (point.y < base.y) base = point;
        if (point.y > top.y) top = point;
    }
    const direction = new Vector2(top.x - base.x, top.z - base.z).normalize();
    const along = (x: number, z: number) => (x - base.x) * direction.x + (z - base.z) * direction.y;

    // Fit the height against the distance along the fall line.
    const distances = points.map((point) => along(point.x, point.z));
    const meanDistance = distances.reduce((sum, value) => sum + value, 0) / points.length;
    const meanHeight = points.reduce((sum, point) => sum + point.y, 0) / points.length;
    let covariance = 0;
    let variance = 0;
    distances.forEach((distance, index) => {
        covariance += (distance - meanDistance) * (points[index].y - meanHeight);
        variance += (distance - meanDistance) ** 2;
    });
    const slope = covariance / variance;
    const minDistance = Math.min(...distances);
    const maxDistance = Math.max(...distances);

    // The slope stops at the base and at the ridge, so the mountain has a top.
    const plane = (x: number, z: number) => {
        const distance = Math.min(Math.max(along(x, z), minDistance - 20), maxDistance + 14);
        return meanHeight + slope * (distance - meanDistance);
    };

    const residuals = points.map((point) => point.y - plane(point.x, point.z));

    return (x: number, z: number) => {
        let weightSum = 0;
        let residualSum = 0;
        points.forEach((point, index) => {
            const weight = 1 / ((x - point.x) ** 2 + (z - point.z) ** 2 + 400);
            weightSum += weight;
            residualSum += weight * residuals[index];
        });
        const bump =
            1.4 * Math.sin(x * 0.19) * Math.cos(z * 0.23) + 0.8 * Math.sin(x * 0.06 + z * 0.09);
        return plane(x, z) + residualSum / weightSum - CLEARANCE + bump;
    };
}

export function Terrain() {
    const geometry = useMemo(() => {
        const surface = buildSurface(resort.nodes.map((node) => nodePosition(node)));
        const plane = new PlaneGeometry(SIZE, SIZE, SEGMENTS, SEGMENTS);
        plane.rotateX(-Math.PI / 2);
        plane.translate(centre.x, 0, centre.z);

        const position = plane.attributes.position as BufferAttribute;
        const colours = new Float32Array(position.count * 3);
        for (let index = 0; index < position.count; index += 1) {
            const x = position.getX(index);
            const z = position.getZ(index);
            const height = surface(x, z);
            position.setY(index, height);

            // Rock shows on the low ground and on the rough ground.
            const rough = Math.sin(x * 0.29) * Math.cos(z * 0.37);
            const snowy = height > 8 || rough < 0.4;
            const colour = snowy ? SNOW : ROCK;
            colours[index * 3] = colour.r;
            colours[index * 3 + 1] = colour.g;
            colours[index * 3 + 2] = colour.b;
        }
        plane.setAttribute("color", new BufferAttribute(colours, 3));
        plane.computeVertexNormals();
        return plane;
    }, []);

    return (
        <mesh geometry={geometry} name="terrain">
            <meshStandardMaterial vertexColors flatShading roughness={0.95} />
        </mesh>
    );
}
