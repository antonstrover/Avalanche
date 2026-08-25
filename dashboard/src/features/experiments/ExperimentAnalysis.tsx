import { ReliabilityPlot } from "./ReliabilityPlot";
import type { Calibration, ModelReference } from "./calibration";

// The screen reads its values from its properties. The compare and experiment
// endpoints arrive with the experiment runner, and the screen then loads the
// same aggregates the analysis writes.
export function ExperimentAnalysis({
    calibration,
    model,
}: {
    calibration: Calibration | null;
    model: ModelReference | null;
}) {
    if (!calibration) {
        return (
            <section className="experiment-analysis" data-testid="experiment-analysis">
                <p className="eyebrow">Experiment analysis</p>
                <h2>No calibrated model</h2>
                <p>Train and calibrate a monitor to see its reliability.</p>
            </section>
        );
    }
    const insideBudget = calibration.false_alarm_rate <= calibration.false_alarm_budget;
    return (
        <section className="experiment-analysis" data-testid="experiment-analysis">
            <p className="eyebrow">Experiment analysis</p>
            <h2>Monitor calibration</h2>
            <dl className="calibration-values">
                <div>
                    <dt>Brier score</dt>
                    <dd data-testid="brier-score">{calibration.brier_score.toFixed(4)}</dd>
                </div>
                <div>
                    <dt>Threshold</dt>
                    <dd data-testid="calibration-threshold">
                        {calibration.threshold.toPrecision(3)}
                    </dd>
                </div>
                <div>
                    <dt>Temperature</dt>
                    <dd data-testid="calibration-temperature">
                        {calibration.temperature.toPrecision(3)}
                    </dd>
                </div>
                <div>
                    <dt>False alarms</dt>
                    <dd data-testid="false-alarm-rate">
                        {(calibration.false_alarm_rate * 100).toFixed(2)}% of a{" "}
                        {(calibration.false_alarm_budget * 100).toFixed(2)}% budget
                    </dd>
                </div>
            </dl>
            <p className="calibration-budget" data-testid="budget-state">
                {insideBudget
                    ? "The threshold stays inside the declared budget."
                    : "The threshold is outside the declared budget."}
            </p>
            <ReliabilityPlot curve={calibration.reliability_curve} />
            {model && (
                <p data-testid="model-reference">
                    {model.model_kind ?? "no model"} · revision{" "}
                    {model.model_revision ?? "unknown"} · features{" "}
                    {model.feature_version ?? "unknown"}
                </p>
            )}
        </section>
    );
}
