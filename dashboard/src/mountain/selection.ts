// A selected item is a piste, a lift, a building, or a hazard.
// Nothing selected is null.
export type Selection = { kind: "piste" | "lift" | "building" | "hazard"; index: number } | null;

export function selectionLabel(selection: Selection): string {
    return selection ? `${selection.kind}: ${selection.index}` : "nothing selected";
}
