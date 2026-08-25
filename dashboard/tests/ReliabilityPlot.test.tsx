import { afterEach, describe, expect, test } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";

import {
    buildReliabilityOption,
    populatedBins,
    symbolSize,
} from "../src/features/experiments/calibration";
import type { Calibration, ReliabilityCurve } from "../src/features/experiments/calibration";
import { ReliabilityPlot } from "../src/features/experiments/ReliabilityPlot";
import { ExperimentAnalysis } from "../src/features/experiments/ExperimentAnalysis";
import fixture from "./fixtures/calibration.json";

const calibration = fixture as Calibration;
const curve = calibration.reliability_curve;

afterEach(cleanup);

describe("the reliability option", () => {
    test("draws the perfect line and the observed points", () => {
        const option = buildReliabilityOption(curve);
        const series = option.series as Array<{ id: string; data: number[][] }>;

        expect(series).toHaveLength(2);
        expect(series[0].id).toBe("perfect");
        expect(series[0].data).toEqual([
            [0, 0],
            [1, 1],
        ]);
        expect(series[1].id).toBe("observed");
        expect(series[1].data).toHaveLength(populatedBins(curve).length);
    });

    test("pairs each predicted value with its observed frequency", () => {
        const option = buildReliabilityOption(curve);
        const series = option.series as Array<{ data: number[][] }>;
        const first = populatedBins(curve)[0];

        expect(series[1].data[0]).toEqual([
            curve.mean_predicted[first],
            curve.observed_frequency[first],
        ]);
    });

    test("holds both axes inside the probability range", () => {
        const option = buildReliabilityOption(curve) as {
            xAxis: { min: number; max: number };
            yAxis: { min: number; max: number };
        };

        expect(option.xAxis.min).toBe(0);
        expect(option.xAxis.max).toBe(1);
        expect(option.yAxis.min).toBe(0);
        expect(option.yAxis.max).toBe(1);
    });

    test("draws no point for an empty bin", () => {
        const empty: ReliabilityCurve = {
            bin_centres: [0.25, 0.75],
            mean_predicted: [0.2, 0.8],
            observed_frequency: [0.1, 0.9],
            counts: [0, 5],
        };

        expect(populatedBins(empty)).toEqual([1]);
        const series = buildReliabilityOption(empty).series as Array<{ data: number[][] }>;
        expect(series[1].data).toEqual([[0.8, 0.9]]);
    });

    test("scales a symbol by its bin count", () => {
        expect(symbolSize(100, 100)).toBeGreaterThan(symbolSize(1, 100));
        expect(symbolSize(0, 0)).toBeGreaterThan(0);
    });
});

describe("the reliability plot", () => {
    test("renders from a fixed calibration fixture", () => {
        render(<ReliabilityPlot curve={curve} />);

        expect(screen.getByTestId("reliability-plot")).toBeInTheDocument();
        expect(screen.getByTestId("reliability-bins")).toHaveTextContent(
            `${populatedBins(curve).length} populated bins of ${curve.counts.length}`,
        );
        // The SVG renderer draws real elements, so the test reads the result
        // and not only the option it asked for.
        expect(
            screen.getByTestId("reliability-canvas").querySelector("svg"),
        ).not.toBeNull();
    });

    test("names the populated bins for a screen reader", () => {
        render(<ReliabilityPlot curve={curve} />);

        expect(screen.getByRole("img")).toHaveAccessibleName(
            `Reliability plot with ${populatedBins(curve).length} populated bins`,
        );
    });
});

describe("the experiment analysis screen", () => {
    test("shows the calibration values", () => {
        render(<ExperimentAnalysis calibration={calibration} model={null} />);

        expect(screen.getByTestId("brier-score")).toHaveTextContent(
            calibration.brier_score.toFixed(4),
        );
        expect(screen.getByTestId("false-alarm-rate")).toHaveTextContent("5.00%");
        expect(screen.getByTestId("budget-state")).toHaveTextContent("inside");
    });

    test("says when no model is calibrated", () => {
        render(<ExperimentAnalysis calibration={null} model={null} />);

        expect(screen.getByTestId("experiment-analysis")).toHaveTextContent(
            "No calibrated model",
        );
    });

    test("names the model behind the calibration", () => {
        render(
            <ExperimentAnalysis
                calibration={calibration}
                model={{
                    model_kind: "perceptron",
                    model_path: "outputs/models/monitor-perceptron.pt",
                    model_revision: "abc123",
                    feature_version: 1,
                }}
            />,
        );

        expect(screen.getByTestId("model-reference")).toHaveTextContent("perceptron");
        expect(screen.getByTestId("model-reference")).toHaveTextContent("abc123");
    });
});
