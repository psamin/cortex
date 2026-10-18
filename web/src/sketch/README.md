# Sketch kit

The Kinetic hero's isometric technical-sketch look, as plain SVG and CSS for React (no WebGL).
Everything is monochrome line art: 1 px ink outlines (`vector-effect: non-scaling-stroke`), flat toon
faces (`#FFFFFF` top, `#EDEDED` left, `#D9D9D9` right, `#BDBDBD` deep), dashed construction lines, and
mono callouts. The projection is true isometric (30°), the hero camera's angle. Dark mode swaps in
light ink on dark grey faces through the CSS variables in `sketch.css` (`[data-theme="dark"]`).

```ts
import { IsoIcon, IsoBox, IsoPlate, Callout, Dimension, IsoGrid, DashedOrbit, SketchScene } from "../sketch";
```

Importing the kit also loads `sketch.css`.

## Components

| Component | Use |
|---|---|
| `<IsoIcon name size? accent? title?>` | A 24×24 isometric glyph. Glyphs: `cube arm bottle camera dataset gpu chart loop shield warning check`. Aliases by tool kind: `robot→arm`, `train→gpu`, `evaluate→chart`, `approve→shield`, `improve→loop`, `demonstrate→camera`, `object→cube`. `ISO_ICON_NAMES` lists them all. |
| `<IsoBox size={[w,d,h]} scale? lift? accent?>` | An extruded box that sizes its own SVG. `lift` raises it (world units) over a dashed footprint, for the exploded-layers look; the lift animates. |
| `<IsoPlate w d h …>` | `IsoBox` preset for a thin slab. |
| `<Callout x y dx dy run label value? accent?>` | An anchor node, a leader line and a mono label. Place it in a `relative` parent; `x`/`y` take px or CSS lengths. Negative `dx` flips the label to the left. |
| `<Dimension value orientation="h"\|"v">` | `|—— 240 MM ——|`: a dimension line with end ticks that fills its container. |
| `<IsoGrid cell? fade?>` | A faint 30°/150° grid that fills its positioned parent. |
| `<DashedOrbit r flat? ticks? axes?>` | A dashed reach circle with yaw ticks; isometric (an ellipse) unless `flat`. Centred by default. |
| `<SketchScene label value width>` | The empty-state drawing: an arm and a cube on a plate, a dashed orbit and one callout. |

Lower-level helpers: `iso(p, scale, origin)`, `poly`, `line`, `boxFaces`, `circle`, `Box` and `Cyl` (for use inside your own `<svg className="sk-svg">`).

## CSS classes

| Class | Effect |
|---|---|
| `.sk-depth` | Gives a card a 4 px isometric extrusion on the bottom-right edge: two skewed hairline side faces in toon greys, not a shadow. Needs a visible overflow on the card. `.sk-depth-lg` makes it 6 px. |
| `.sk-depth.sk-hover` | The card lifts 2 px up and left on hover or focus-within, and its faces grow, so the base stays put. It uses `top`/`left`, not `transform`, so fixed-position children keep working. |
| `.sk-pill` + `.sk-pill-primary` / `-secondary` / `-danger` | A pill button with an offset base plate under it, which compresses on press. Set the height and padding yourself. |
| `.sk-svg`, `.sk-f1…f4`, `.sk-construction`, `.sk-leader`, `.sk-node`, `.sk-accent` | Line-art primitives for custom drawings. |

All motion respects `prefers-reduced-motion`.
