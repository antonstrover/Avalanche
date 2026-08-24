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
    const choices = {
        mountain: options.mountains,
        scenario: options.scenarios,
        controller: options.controllers,
        monitor: options.monitors,
    };
    const changeChoice = (name: ChoiceName, value: string) => {
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
                <pre data-testid="resolved-config">{JSON.stringify(resolved, null, 2)}</pre>
            ) : (
                <p>Resolving the selected configuration.</p>
            )}
        </section>
    );
}
