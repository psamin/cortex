import { useState } from "react";

/** The human name recorded as a training approver, remembered per browser. */
export function useApprover() {
  const [name, setName] = useState(() => {
    try {
      return localStorage.getItem("cortex.approver") ?? "";
    } catch {
      return "";
    }
  });
  const save = (n: string) => {
    setName(n);
    try {
      localStorage.setItem("cortex.approver", n);
    } catch {
      /* storage unavailable */
    }
  };
  return [name, save] as const;
}
