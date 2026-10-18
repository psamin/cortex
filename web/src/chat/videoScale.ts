/** Videos narrower than this are low resolution (older episodes' 128x96 robot cameras, served at 2x). */
const SMALL_VIDEO_W = 640;
/** How far the viewer may enlarge a low-resolution video: stretched further it only looks blurrier. */
export const MAX_UPSCALE = 2;

/** The tallest a video should be drawn (CSS px), or null for no limit. */
export function maxDisplayHeight(videoWidth: number, videoHeight: number): number | null {
  return videoWidth > 0 && videoWidth < SMALL_VIDEO_W ? videoHeight * MAX_UPSCALE : null;
}
