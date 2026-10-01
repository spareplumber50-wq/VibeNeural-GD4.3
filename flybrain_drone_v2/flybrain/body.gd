class_name Body
extends RefCounted
## Anything the fly brain can control: a drone, a quadruped, a biped.
## The brain only sees sense() and only acts through step().

const CTRL_DT := 0.02          # brain / control loop: 50 Hz
const RAYS := 8                # range-finder whiskers around the body
const RAY_RANGE := 3.0

var crashed := false
var pos := Vector3.ZERO
var q := Quaternion.IDENTITY

func n_sensors() -> int:
	return 0
func n_outputs() -> int:
	return 0
func body_radius() -> float:
	return 0.2
func spawn_height() -> float:
	return 1.5
func retarget_period() -> float:
	return 4.0
func success_threshold() -> float:
	return 0.7
func reset(_rng: RandomNumberGenerator, _difficulty: float, _obs: Obstacles) -> void:
	pass
func sense(_target: Vector3, _obs: Obstacles) -> PackedFloat32Array:
	return PackedFloat32Array()
func step(_out: PackedFloat32Array, _obs: Obstacles) -> void:
	pass
func reward(_target: Vector3) -> float:
	return 0.0
func make_target(_rng: RandomNumberGenerator, _d: float, _radius: float, _obs: Obstacles) -> Vector3:
	return Vector3.ZERO
func retarget() -> void:
	pass
func reached(_target: Vector3) -> bool:
	return false

## proximity whiskers: 0 = nothing in range, 1 = touching
func ray_sensors(obs: Obstacles) -> PackedFloat32Array:
	var out := PackedFloat32Array()
	out.resize(RAYS)
	if obs == null or obs.count() == 0:
		return out
	for k in RAYS:
		var a := TAU * k / RAYS
		var dir := q * Vector3(sin(a), 0.0, -cos(a))
		out[k] = 1.0 - minf(obs.ray(pos, dir, RAY_RANGE), RAY_RANGE) / RAY_RANGE
	return out
