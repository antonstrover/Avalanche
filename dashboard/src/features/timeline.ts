import type { TimelineEvent } from "../workers/live-frame";

const precursorNames: Record<string, { eventType: string; label: string }> = {
    early_indicator: {
        eventType: "density_warning",
        label: "density warning",
    },
    true_harm: {
        eventType: "capacity_exposure",
        label: "capacity exposure",
    },
    density_warning: {
        eventType: "density_warning",
        label: "density warning",
    },
    capacity_exposure: {
        eventType: "capacity_exposure",
        label: "capacity exposure",
    },
};

export function migrateTimelineEvent(event: TimelineEvent): TimelineEvent {
    const precursor = precursorNames[event.event_type];
    if (!precursor) return event;
    const separator = event.event_id.indexOf(":");
    const suffix = separator < 0 ? "" : event.event_id.slice(separator);
    return {
        ...event,
        event_id: `${precursor.eventType}${suffix}`,
        event_type: precursor.eventType,
        label: precursor.label,
    };
}

export function mergeTimeline(
    current: TimelineEvent[],
    incoming: TimelineEvent[],
): TimelineEvent[] {
    const merged = new Map(
        current.map(migrateTimelineEvent).map((event) => [event.event_id, event]),
    );
    for (const value of incoming) {
        const event = migrateTimelineEvent(value);
        merged.set(event.event_id, event);
    }
    return [...merged.values()]
        .sort(
            (left, right) =>
                left.start_time_seconds - right.start_time_seconds ||
                left.event_id.localeCompare(right.event_id),
        )
        .slice(-64);
}
