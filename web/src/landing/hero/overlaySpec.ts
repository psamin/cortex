/** Callout leaders drawn from joints (screen-space offsets in CSS px). */
export interface CalloutSpec {
  id: string;
  /** Joint index, or -1 for the TCP. */
  joint: number;
  label: string;
  dx: number;
  dy: number;
  run: number;
}

export const CALLOUTS: CalloutSpec[] = [
  { id: "base", joint: 0, label: "J01 / BASE", dx: 64, dy: 54, run: 30 },
  { id: "shoulder", joint: 1, label: "J02 / SHOULDER", dx: 92, dy: -26, run: 30 },
  { id: "elbow", joint: 2, label: "J03 / ELBOW", dx: 58, dy: -58, run: 30 },
  { id: "tcp", joint: -1, label: "TCP", dx: 30, dy: 70, run: 28 },
];
