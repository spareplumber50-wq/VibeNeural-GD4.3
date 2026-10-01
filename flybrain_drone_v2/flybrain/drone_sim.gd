class_name DroneSim
extends RefCounted
## Rigid-body multirotor. Custom integrator (not Godot physics) so training can
## run much faster than real time on worker threads.

const G := 9.81
const WN := 14.0        # onboard attitude loop natural frequency (rad/s)
const ZETA := 0.8

var cfg: DroneConfig
var pos := Vector3.ZERO
var vel := Vector3.ZERO
var q := Quaternion.IDENTITY
var omega := Vector3.ZERO            # body-frame angular velocity
var thrust := PackedFloat32Array()
var motor_pos: Array[Vector3] = []
var motor_dir := PackedFloat32Array()
var inertia := Vector3.ONE
var _sx := 1.0     # sum of squared normalised motor x / z offsets (mixer effectiveness)
var _sz := 1.0
var mass := 1.0
var wind_vec := Vector3.ZERO
var gust_state := Vector3.ZERO
var crashed := false
var rng := RandomNumberGenerator.new()

func _init(c: DroneConfig, seed_: int) -> void:
	cfg = c
	rng.seed = seed_
	mass = c.total_mass()
	thrust.resize(c.motor_count)
	motor_dir.resize(c.motor_count)
	var ib := 0.4 * c.body_mass * 0.05 * 0.05
	var ix := ib
	var iy := ib
	var iz := ib
	for i in c.motor_count:
		var a := TAU * (i + 0.5) / c.motor_count
		var p := Vector3(cos(a), 0.0, sin(a)) * c.arm_length
		motor_pos.append(p)
		motor_dir[i] = 1.0 if i % 2 == 0 else -1.0
		ix += c.motor_mass * p.z * p.z
		iz += c.motor_mass * p.x * p.x
		iy += c.motor_mass * (p.x * p.x + p.z * p.z)
	inertia = Vector3(ix, iy, iz)
	_sx = 0.0
	_sz = 0.0
	for p in motor_pos:
		_sx += (p.x / c.arm_length) * (p.x / c.arm_length)
		_sz += (p.z / c.arm_length) * (p.z / c.arm_length)

func reset(start: Vector3, tilt: float, speed: float, r: RandomNumberGenerator) -> void:
	pos = start
	var ang := r.randf() * TAU
	q = Quaternion(Vector3(cos(ang), 0.0, sin(ang)), tilt * r.randf())
	vel = Vector3(r.randfn(), r.randfn() * 0.3, r.randfn()).normalized() * speed * r.randf()
	omega = Vector3.ZERO
	thrust.fill(mass * G / cfg.motor_count)
	var wa := r.randf() * TAU
	wind_vec = Vector3(cos(wa), 0.0, sin(wa)) * cfg.wind
	gust_state = Vector3.ZERO
	crashed = false

## Onboard flight controller. stick = [accel_x, accel_z, yaw_rate, climb], each -1..1.
## accel sticks tilt the thrust vector (body-yaw frame); gains auto-scale to the drone.
func stabilize(stick: PackedFloat32Array) -> PackedFloat32Array:
	var b := Basis(q)
	var hx := Vector3(b.x.x, 0.0, b.x.z)
	var yawq := Quaternion.IDENTITY
	if hx.length() > 0.05:
		yawq = Quaternion(Vector3.UP, atan2(-hx.z, hx.x))
	var mt := tan(cfg.max_tilt)
	var up_des_w := (yawq * Vector3(stick[0] * mt, 1.0, stick[1] * mt)).normalized()
	var inv := q.inverse()
	var up_b := inv * Vector3.UP
	var delta := Vector3.UP.cross(inv * up_des_w)
	var tau_x := inertia.x * (WN * WN * delta.x - 2.0 * ZETA * WN * omega.x)
	var tau_z := inertia.z * (WN * WN * delta.z - 2.0 * ZETA * WN * omega.z)
	var tau_y := inertia.y * 8.0 * (stick[2] * 2.0 - omega.y)
	var u_x := tau_x / (cfg.max_thrust * cfg.arm_length * maxf(_sz, 0.01))
	var u_z := tau_z / (cfg.max_thrust * cfg.arm_length * maxf(_sx, 0.01))
	var u_y := tau_y / (cfg.yaw_coeff * cfg.max_thrust * cfg.motor_count)
	var thr := cfg.hover_fraction() * (1.0 + 0.5 * stick[3]) / maxf(up_b.y, 0.5)
	var cmd := PackedFloat32Array()
	cmd.resize(cfg.motor_count)
	for i in cfg.motor_count:
		var pn := motor_pos[i] / cfg.arm_length
		cmd[i] = clampf(thr - u_x * pn.z + u_z * pn.x + u_y * motor_dir[i], 0.0, 1.0)
	return cmd

func step(cmd: PackedFloat32Array, dt: float, substeps: int = 4) -> void:
	var h := dt / substeps
	for _s in substeps:
		var total := 0.0
		var torque := Vector3.ZERO
		for i in cfg.motor_count:
			var target_t := clampf(cmd[i], 0.0, 1.0) * cfg.max_thrust
			thrust[i] += (target_t - thrust[i]) * minf(h / cfg.motor_tau, 1.0)
			var t := thrust[i]
			total += t
			var p := motor_pos[i]
			torque += Vector3(-p.z * t, motor_dir[i] * cfg.yaw_coeff * t, p.x * t)
		torque -= omega * cfg.ang_damping
		var air := wind_vec + gust_state
		var f := (q * Vector3(0.0, total, 0.0)) + Vector3(0.0, -mass * G, 0.0) + cfg.drag * (air - vel)
		vel += f / mass * h
		pos += vel * h
		var iw := inertia * omega
		omega += (torque - omega.cross(iw)) / inertia * h
		var a := omega.length() * h
		if a > 1e-9:
			q = (q * Quaternion(omega.normalized(), a)).normalized()
	if cfg.gust > 0.0:
		var k := 2.0 * dt / 1.5
		gust_state = gust_state * (1.0 - dt / 1.5) + Vector3(rng.randfn(), rng.randfn() * 0.3, rng.randfn()) * cfg.gust * sqrt(k)
	if pos.y <= 0.0 or (q * Vector3.UP).y < 0.0 or absf(pos.x) > 15.0 or absf(pos.z) > 15.0 or pos.y > 30.0 \
			or is_nan(pos.x) or is_nan(omega.x):
		crashed = true
		pos.y = maxf(pos.y, 0.0)

## 12 normalised sensors. Loosely fly-inspired:
## [0..2] target direction (vision) [3..5] body velocity (optic flow)
## [6..8] up-vector (ocelli / horizon) [9..11] angular rate (halteres)
func sense(target: Vector3) -> PackedFloat32Array:
	var inv := q.inverse()
	var rel := inv * (target - pos)
	var v := inv * vel
	var up := inv * Vector3.UP
	var s := PackedFloat32Array([rel.x / 2.0, rel.y / 2.0, rel.z / 2.0,
		v.x / 3.0, v.y / 3.0, v.z / 3.0, up.x, up.y, up.z,
		omega.x / 5.0, omega.y / 5.0, omega.z / 5.0])
	for i in 12:
		s[i] = clampf(s[i] + rng.randfn(0.0, cfg.sensor_noise), -1.0, 1.0)
	return s
