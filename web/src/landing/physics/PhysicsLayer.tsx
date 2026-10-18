import { Physics } from "@react-three/rapier";
import { GraspableCube } from "./GraspableCube";

/** Rapier world + cube, split into its own chunk so the arm renders before the physics wasm loads. */
export default function PhysicsLayer({ debug }: { debug: boolean }) {
  return (
    <Physics timeStep={1 / 60} interpolate debug={debug}>
      <GraspableCube />
    </Physics>
  );
}
