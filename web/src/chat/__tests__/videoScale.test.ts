import { describe, expect, it } from "vitest";
import { MAX_UPSCALE, maxDisplayHeight } from "../videoScale";

describe("maxDisplayHeight", () => {
  it("limits low-resolution videos to 2x their size", () => {
    expect(maxDisplayHeight(256, 192)).toBe(192 * MAX_UPSCALE); // an older episode's 128x96 camera, served at 2x
  });

  it("lets sharp videos fill the stage", () => {
    expect(maxDisplayHeight(640, 480)).toBeNull(); // the recorder's display copy of a robot camera
    expect(maxDisplayHeight(1280, 960)).toBeNull(); // the studio view
  });

  it("does not limit a video before its size is known", () => {
    expect(maxDisplayHeight(0, 0)).toBeNull();
  });
});
