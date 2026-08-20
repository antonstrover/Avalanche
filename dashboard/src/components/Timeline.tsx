import type { TimelineEvent, WeatherState } from "../workers/live-frame";

function time(value: number | null): string {
    if (value === null) return "ongoing";
    const minutes = Math.floor(value / 60);
    const seconds = Math.floor(value % 60);
    return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

export function Timeline({
    events,
    weather,
}: {
    events: TimelineEvent[];
    weather: WeatherState;
}) {
    return (
        <aside className="timeline" aria-labelledby="timeline-title">
            <div className="timeline-heading">
                <div>
                    <p className="eyebrow">Live conditions</p>
                    <h2 id="timeline-title">Event timeline</h2>
                </div>
                <dl className="weather-summary" aria-label="Current weather">
                    <div>
                        <dt>Wind</dt>
                        <dd>{weather.wind.toFixed(1)} m/s</dd>
                    </div>
                    <div>
                        <dt>Visibility</dt>
                        <dd>{Math.round(weather.visibility)} m</dd>
                    </div>
                    <div>
                        <dt>Snow</dt>
                        <dd>{weather.snowfall.toFixed(1)}</dd>
                    </div>
                    <div>
                        <dt>Temp.</dt>
                        <dd>{weather.temperature.toFixed(1)} °C</dd>
                    </div>
                </dl>
            </div>
            {events.length === 0 ? (
                <p className="timeline-empty">No material events yet.</p>
            ) : (
                <ol className="timeline-list" aria-live="polite">
                    {[...events].reverse().map((event) => (
                        <li
                            key={event.event_id}
                            data-event-id={event.event_id}
                            data-event-type={event.event_type}
                            className={`timeline-event severity-${event.severity}`}
                        >
                            <span className="timeline-marker" aria-hidden="true" />
                            <span className="timeline-time">{time(event.start_time_seconds)}</span>
                            <span>
                                <strong>{event.label}</strong>
                                <span className="timeline-target">{event.target}</span>
                                <span className="timeline-severity">
                                    {event.severity} severity
                                </span>
                            </span>
                            <span className="timeline-end">Until {time(event.end_time_seconds)}</span>
                        </li>
                    ))}
                </ol>
            )}
        </aside>
    );
}
