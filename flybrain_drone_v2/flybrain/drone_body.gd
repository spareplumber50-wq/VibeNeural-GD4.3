class_name DroneBody
extends Body

const VIS := 1.6               # the drone is drawn this much bigger than its physics size

var cfg: DroneConfig
var sim: DroneSim

func _init(c: DroneConfig, seed_: int) -> void:
	cfg = c
	sim = DroneSim.new(c, seed_)
	_sync()

func n_sensors() -> int:
	return 12 + RAYS
func n_outputs() -> int:
	return cfg.channels()
func body_radius() -> float:
	return cfg.arm_length * VIS + 0.04

func _sync() -> void:
	pos = sim.pos
	q = sim.q

func reset(rng: RandomNumberGenerator, d: float, _obs: Obstacles) -> void:
	sim.reset(Vector3(0.0, spawn_height(), 0.0), 0.05 + 0.25 * d, 0.1 + 0.5 * d, rng)
	crashed = false
	_sync()

func sense(target: Vector3, obs: Obstacles) -> PackedFloat32Array:
	var s := sim.sense(target)
	s.append_array(ray_sensors(obs))
	return s

func step(out: PackedFloat32Array, obs: Obstacles) -> void:
	var cmd: PackedFloat32Array
	if cfg.assist:
		cmd = sim.stabilize(out)
	else:
		cmd = PackedFloat32Array()
		cmd.resize(cfg.motor_count)
		var hover := cfg.hover_fraction()
		for i in cfg.motor_count:
			cmd[i] = clampf(hover + 0.15 * out[i], 0.0, 1.0)
	sim.step(cmd, CTRL_DT)
	_sync()
	crashed = sim.crashed or (obs != null and obs.hit(sim.pos, body_radius()))

func reward(target: Vector3) -> float:
	return exp(-0.8 * (target - sim.pos).length()) - 0.005 * sim.omega.length_squared()

func make_target(rng: RandomNumberGenerator, d: float, radius: float, obs: Obstacles) -> Vector3:
	var start := Vector3(0.0, spawn_height(), 0.0)
	var r := d * radius
	if r <= 0.0:
		return start
	var t := start
	for _i in 30:
		t = Vector3(start.x + rng.randf_range(-r, r), clampf(start.y + rng.randf_range(-r, r) * 0.5, 0.6, 4.0), start.z + rng.randf_range(-r, r))
		if obs == null or not obs.hit(t, 0.4):
			break
	return t
