# FlyBrain (Godot 4.3+): a fruit-fly connectome that flies drones and walks

Open the folder in Godot 4.3+ (first open imports the project), press F5, pick an example, press **Start training**.

## What's new
- **GPU brain**: every generation, all candidates x flights advance in lock-step; the whole connectome for all of them is
  one compute-shader dispatch per 20 ms tick (`brain_batch.gd`). Bodies are simulated on CPU worker threads. Falls back to CPU
  threads automatically if there is no compute device (needs the Forward+ or Mobile renderer). Raise *Neurons*, *Synapses per
  neuron*, *Population* and *Flights per candidate* to put more load on the GPU (up to 30 000 neurons in the UI).
- **Scene editor**: choose a mouse tool and left-click the ground to add pillars, walls, spheres, delete the nearest obstacle,
  or place the target. Right-drag orbits. 8 range-finder whiskers let the brain sense obstacles; touching one ends the flight.
  "Random obstacles in training" adds a fresh random set every flight so the brain generalises.
- **8 examples** (Scene > Example): hover and chase, pillar slalom, narrow gate in wind, random forest, hexacopter in crosswind,
  quadruped, quadruped obstacle field, biped.
- **Quadruped and biped** (Body > Creature): torso + stiff position-controlled legs (hip swing, hip splay, leg extension per leg),
  penalty ground contact with friction, foot contact lights green. The brain gets torso attitude/velocity, target direction,
  leg joint state and foot contact; it outputs 3 targets per leg. The biped starts with "training wheels" (upright assist) that
  fade as fitness rises.

## Honest expectations
- Quadruped: in my headless tests it learned to walk ~1.7 m to a target in about 100 generations.
- Biped: experimental. It learns to stand and shuffle toward the target as the training wheels fade, not a clean stride.
- Obstacle courses train slower than open sky; use random obstacles and let the curriculum run.
- A policy only fits its own body type, sensor layout and brain size (Save/Load stores all of that).

## Files
main.gd (UI, 3D, editor) - brain_batch.gd (GPU/CPU connectome) - trainer.gd (threaded evolution strategy) - runner.gd (live run)
body.gd, drone_body.gd, walker_body.gd, body_spec.gd, drone_sim.gd, drone_config.gd, walker_config.gd (creatures)
obstacles.gd - examples.gd - fly_brain.gd (connectome: synthetic or CSV import)

Real connectome import: see the earlier instructions (CSV edge list `pre_id,post_id,weight`, optional `neurons.csv`).
