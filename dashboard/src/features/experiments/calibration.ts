// The calibration values a run records beside its model.
// The shapes match `ReliabilityCurve.as_dict` and `Calibration.as_dict`
// in `src/avalanche/monitors/calibration.py`.
// The frontend defines no research metric of its own.

import type { EChartsOption } from "echarts";

export type ReliabilityCurve = {
    bin_centres: number[];
    mean_predicted: number[];
    observed_frequency: number[];
    counts: number[];
};

export type Calibration = {
    temperature: number;
    threshold: number;
    false_alarm_budget: number;
    false_alarm_rate: number;
    brier_score: number;
    reliability_curve: ReliabilityCurve;
};

export type ModelReference = {
    model_kind: string | null;
    model_path: string | null;
    model_revision: string | null;
    feature_version?: number | null;
    threshold?: number | null;
    temperature?: number | null;
};

const MAXIMUM_SYMBOL = 26;
const MINIMUM_SYMBOL = 4;

// A bin with no row must not draw a point. Its predicted value is zero, and a
// zero would look like a calibrated bin at the origin.
export function populatedBins(curve: ReliabilityCurve): number[] {
    return curve.counts.map((_, index) => index).filter((index) => curve.counts[index] > 0);
}

export function symbolSize(count: number, largest: number): number {
    if (largest <= 0) return MINIMUM_SYMBOL;
    const share = Math.sqrt(count / largest);
    return MINIMUM_SYMBOL + share * (MAXIMUM_SYMBOL - MINIMUM_SYMBOL);
}

export function buildReliabilityOption(curve: ReliabilityCurve): EChartsOption {
    const bins = populatedBins(curve);
    const largest = Math.max(0, ...curve.counts);
    const observed = bins.map((index) => [
        curve.mean_predicted[index],
        curve.observed_frequency[index],
    ]);
    const sizes = bins.map((index) => symbolSize(curve.counts[index], largest));
    return {
        animation: false,
        grid: { left: 56, right: 24, top: 32, bottom: 44 },
        xAxis: {
            type: "value",
            min: 0,
            max: 1,
            name: "Predicted probability",
            nameLocation: "middle",
            nameGap: 28,
        },
        yAxis: {
            type: "value",
            min: 0,
            max: 1,
            name: "Observed frequency",
            nameLocation: "middle",
            nameGap: 40,
        },
        tooltip: { trigger: "item" },
        series: [
            {
                id: "perfect",
                name: "Perfect calibration",
                type: "line",
                data: [
                    [0, 0],
                    [1, 1],
                ],
                showSymbol: false,
                lineStyle: { type: "dashed" },
            },
            {
                id: "observed",
                name: "Observed",
                type: "line",
                data: observed,
                symbolSize: (_value: unknown, parameters: { dataIndex: number }) =>
                    sizes[parameters.dataIndex],
            },
        ],
    };
}
