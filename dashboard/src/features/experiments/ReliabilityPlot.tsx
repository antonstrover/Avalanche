import { useEffect, useRef } from "react";
import * as echarts from "echarts";

import { buildReliabilityOption, populatedBins } from "./calibration";
import type { ReliabilityCurve } from "./calibration";

// A thin wrapper keeps the chart out of React. A second dependency for that
// alone is not worth its size.
export function ReliabilityPlot({
    curve,
    height = 320,
}: {
    curve: ReliabilityCurve;
    height?: number;
}) {
    const container = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!container.current) return;
        // The SVG renderer keeps the chart out of a canvas. It stays crisp on
        // a high-density display and it renders in the test environment.
        const chart = echarts.init(container.current, null, { renderer: "svg" });
        chart.setOption(buildReliabilityOption(curve));
        const resize = () => chart.resize();
        window.addEventListener("resize", resize);
        return () => {
            window.removeEventListener("resize", resize);
            chart.dispose();
        };
    }, [curve]);

    const populated = populatedBins(curve).length;
    return (
        <figure className="reliability-plot" data-testid="reliability-plot">
            <div
                ref={container}
                style={{ width: "100%", height }}
                data-testid="reliability-canvas"
                role="img"
                aria-label={`Reliability plot with ${populated} populated bins`}
            />
            <figcaption data-testid="reliability-bins">
                {populated} populated bins of {curve.counts.length}
            </figcaption>
        </figure>
    );
}
