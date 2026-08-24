import type { ConfigOptionsResponse } from "../../api/client";

export function SessionSetup({
    options,
    failed,
}: {
    options: ConfigOptionsResponse | null;
    failed: boolean;
}) {
    if (failed) {
        return <p role="alert">The configuration choices could not load.</p>;
    }
    if (!options) {
        return <p>Loading the configuration choices.</p>;
    }
    return (
        <section className="session-setup" data-testid="session-setup">
            <h2>Live configuration</h2>
            <dl>
                <div><dt>Mountains</dt><dd>{options.mountains.length}</dd></div>
                <div><dt>Scenarios</dt><dd>{options.scenarios.length}</dd></div>
                <div><dt>Controllers</dt><dd>{options.controllers.length}</dd></div>
                <div><dt>Monitors</dt><dd>{options.monitors.length}</dd></div>
            </dl>
        </section>
    );
}
