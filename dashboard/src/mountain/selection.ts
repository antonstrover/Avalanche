// A selected item is a piste, a lift, or a building. Nothing selected is null.
export type Selection = { kind: "piste" | "lift" | "building"; index: number } | null;

export function selectionLabel(selection: Selection): string {
    return selection ? `${selection.kind}: ${selection.index}` : "nothing selected";
}
