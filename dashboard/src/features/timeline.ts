import type { TimelineEvent } from "../workers/live-frame";

export function mergeTimeline(
    current: TimelineEvent[],
    incoming: TimelineEvent[],
): TimelineEvent[] {
    const merged = new Map(current.map((event) => [event.event_id, event]));
    for (const event of incoming) merged.set(event.event_id, event);
    return [...merged.values()]
        .sort(
            (left, right) =>
                left.start_time_seconds - right.start_time_seconds ||
                left.event_id.localeCompare(right.event_id),
        )
        .slice(-64);
}
