import type {
    ConfigOptionsResponse,
    LiveConfigSelection,
    ResolvedLiveConfig,
} from "../../api/client";

type ChoiceName = "mountain" | "scenario" | "controller" | "monitor";

export function SessionSetup({
    options,
    selection,
    resolved,
    failed,
    onChange,
}: {
    options: ConfigOptionsResponse | null;
    selection: LiveConfigSelection;
    resolved: ResolvedLiveConfig | null;
    failed: boolean;
    onChange: (selection: LiveConfigSelection) => void;
}) {
    if (failed) {
        return <p role="alert">The configuration choices could not load.</p>;
    }
    if (!options) {
        return <p>Loading the configuration choices.</p>;
    }
    const compatibleControllers = options.controllers.filter((option) =>
        option.compatible_mountain_ids.includes(selection.mountain),
    );
    const choices = {
        mountain: options.mountains,
        scenario: options.scenarios,
        controller: compatibleControllers,
        monitor: options.monitors,
    };
    const changeChoice = (name: ChoiceName, value: string) => {
        if (name === "mountain") {
            const compatible = options.controllers.filter((option) =>
                option.compatible_mountain_ids.includes(value),
            );
            const currentIsCompatible = compatible.some(
                (option) => option.id === selection.controller,
            );
            const preferredId = value === "medium-resort" ? "honest" : `${value}/honest`;
            const preferred = compatible.find((option) => option.id === preferredId);
            const honest = compatible.find(
                (option) => option.controller.kind === "honest",
            );
            const controller = currentIsCompatible
                ? selection.controller
                : (preferred ?? honest ?? compatible[0])?.id;
            if (controller) {
                onChange({ ...selection, mountain: value, controller });
            }
            return;
        }
        onChange({ ...selection, [name]: value });
    };
    return (
        <section className="session-setup" data-testid="session-setup">
            <h2>Live configuration</h2>
            <div className="configuration-fields">
                {(Object.keys(choices) as ChoiceName[]).map((name) => (
                    <label key={name}>
                        {name}
                        <select
                            value={selection[name]}
                            onChange={(event) => changeChoice(name, event.target.value)}
                        >
                            {choices[name].map((option) => (
                                <option key={option.id} value={option.id}>
                                    {option.label}
                                </option>
                            ))}
                        </select>
                    </label>
                ))}
                <label>
                    Seed
                    <input
                        type="number"
                        value={selection.seed}
                        onChange={(event) =>
                            onChange({ ...selection, seed: Number(event.target.value) })
                        }
                    />
                </label>
                <label>
                    Skier count
                    <input
                        type="number"
                        min="1"
                        max="10000"
                        value={selection.skier_count}
                        onChange={(event) =>
                            onChange({
                                ...selection,
                                skier_count: Number(event.target.value),
                            })
                        }
                    />
                </label>
            </div>
            <h3>Resolved configuration</h3>
            {resolved ? (
                <div data-testid="resolved-config">
                    <dl className="configuration-summary">
                        <div>
                            <dt>Mountain</dt>
                            <dd>{resolved.mountain.name}</dd>
                        </div>
                        <div>
                            <dt>Scenario</dt>
                            <dd>{resolved.scenario.name}</dd>
                        </div>
                        <div>
                            <dt>Controller</dt>
                            <dd>{resolved.controller.kind}</dd>
                        </div>
                        <div>
                            <dt>Monitor</dt>
                            <dd>{resolved.monitor.kind}</dd>
                        </div>
                        <div>
                            <dt>Seed</dt>
                            <dd>{resolved.seed}</dd>
                        </div>
                        <div>
                            <dt>Skier count</dt>
                            <dd>{resolved.population.skier_count}</dd>
                        </div>
                    </dl>
                    <details className="configuration-details">
                        <summary>View the full configuration</summary>
                        <pre>{JSON.stringify(resolved, null, 2)}</pre>
                    </details>
                </div>
            ) : (
                <p>Resolving the selected configuration.</p>
            )}
        </section>
    );
}
