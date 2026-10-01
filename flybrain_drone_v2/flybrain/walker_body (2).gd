class_name WalkerBody
extends Body
## Custom legged-robot physics (fast, runs on worker threads): 6-DOF torso, massless
## position-controlled legs, penalty ground contact with regularised Coulomb friction.
## Sweeping a leg backwards while its foot grips the ground pushes the torso forward.

const SUB := 5
const G := 9.81

var cfg: WalkerConfig
var rng := RandomNumberGenerator.new()
var vel := Vector3.ZERO
var omega := Vector3.ZERO                 # body frame
var legs := 0
var hip: Array[Vector3] = []
var side := PackedFloat32Array()
var phi := PackedFloat32Array()
var psi := PackedFloat32Array()
var ell := PackedFloat32Array()
var foot_b: Array[Vector3] = []           # foot centre in body frame
var contact := PackedFloat32Array()
var pts: Array[Vector3] = []
var mass := 1.0
var inertia := Vector3.ONE
var k_pt := 1000.0
var c_pt := 30.0
var rest_y := 0.3
var prev_dist := -1.0
var harness := 0.0          # training wheels: upright torque (+ weight support for biped), fades with difficulty

func _init(c: WalkerConfig, seed_: int) -> void:
	cfg = c
	rng.seed = seed_
	legs = c.legs()
	hip = c.hips()
	mass = c.body_mass
	side.resize(legs)
	phi.resize(legs)
	psi.resize(legs)
	ell.resize(legs)
	contact.resize(legs)
	for i in legs:
		side[i] = 1.0 if hip[i].x > 0.0 else -1.0
		foot_b.append(Vector3.ZERO)
	var fh := c.foot_half()
	if fh == Vector2.ZERO:
		pts.append(Vector3.ZERO)
	else:
		for sx in [-1.0, 1.0]:
			for sz in [-1.0, 1.0]:
				pts.append(Vector3(sx * fh.x, 0.0, sz * fh.y))
	var m := c.body_mass
	inertia = Vector3(m / 12.0 * (c.body_height * c.body_height + c.body_length * c.body_length),
		m / 12.0 * (c.body_width * c.body_width + c.body_length * c.body_length),
		m / 12.0 * (c.body_width * c.body_width + c.body_height * c.body_height))
	var k_leg := m * G / legs / 0.012                      # ~1.2 cm static deflection
	k_pt = k_leg / pts.size()
	c_pt = 2.0 * c.damping_ratio * sqrt(k_leg * m / legs) / pts.size()
	rest_y = c.body_height * 0.5 + c.leg_length * cos(c.rest_splay()) - 0.008
	_pose_rest()

func n_sensors() -> int:
	return 12 + 4 * legs + RAYS
func n_outputs() -> int:
	return 3 * legs
func body_radius() -> float:
	return maxf(cfg.body_width, cfg.body_length) * 0.6
func spawn_height() -> float:
	return rest_y
func retarget_period() -> float:
	return 14.0
func success_threshold() -> float:
	return 0.45

func _leg_dir(p: float, s: float) -> Vector3:
	return Vector3(sin(s), -cos(s) * cos(p), -cos(s) * sin(p))

func _pose_rest() -> void:
	for i in legs:
		phi[i] = 0.0
		psi[i] = side[i] * cfg.rest_splay()
		ell[i] = cfg.leg_length
		foot_b[i] = hip[i] + ell[i] * _leg_dir(phi[i], psi[i])
		contact[i] = 0.0

func reset(r: RandomNumberGenerator, d: float, _obs: Obstacles) -> void:
	rng.seed = r.randi()
	var yaw := r.randf_range(-PI, PI) * d
	pos = Vector3(0.0, rest_y, 0.0)
	q = Quaternion(Vector3.UP, yaw)
	vel = Vector3.ZERO
	omega = Vector3.ZERO
	crashed = false
	hit_obstacle = false
	prev_dist = -1.0
	harness = (1.0 - d) * (1.0 if cfg.kind == "biped" else 0.5)
	_pose_rest()

func step(out: PackedFloat32Array, obs: Obstacles) -> void:
	var h := CTRL_DT / SUB
	var a_j := 1.0 - exp(-h / cfg.motor_tau)
	var mu := cfg.friction
	for _s in SUB:
		var b := Basis(q)
		var force := Vector3(0.0, -mass * G, 0.0)
		var torque_w := Vector3.ZERO
		for i in legs:
			var pc := out[3 * i] * cfg.swing
			var sc := side[i] * (cfg.rest_splay() + out[3 * i + 1] * cfg.splay)
			var lc := cfg.leg_length + out[3 * i + 2] * cfg.leg_range
			phi[i] += (pc - phi[i]) * a_j
			psi[i] += (sc - psi[i]) * a_j
			ell[i] += (lc - ell[i]) * a_j
			var fb := hip[i] + ell[i] * _leg_dir(phi[i], psi[i])
			var fv_b := (fb - foot_b[i]) / h
			foot_b[i] = fb
			contact[i] = 0.0
			for o in pts:
				var pb := fb + o
				var r_w := b * pb
				var pen := -(pos.y + r_w.y)
				if pen > 0.0:
					var vw := vel + b * (omega.cross(pb) + fv_b)
					var fn := maxf(k_pt * pen - c_pt * vw.y, 0.0)
					var vt := Vector3(vw.x, 0.0, vw.z)
					var ft := -mu * fn * vt / (vt.length() + 0.03)
					var f := Vector3(ft.x, fn, ft.z)
					force += f
					torque_w += r_w.cross(f)
					contact[i] = 1.0
		if harness > 0.0:
			torque_w += harness * ((b * Vector3.UP).cross(Vector3.UP) * (mass * G * 0.6) - (b * omega) * (mass * 0.35))
			if legs == 2:
				force.y += harness * 0.6 * mass * G
		vel += force / mass * h
		pos += vel * h
		var tb := b.transposed() * torque_w
		omega += (tb - omega.cross(inertia * omega)) / inertia * h
		omega *= 1.0 - 0.5 * h
		var ang := omega.length() * h
		if ang > 1e-9:
			q = (q * Quaternion(omega.normalized(), ang)).normalized()
	var up_y := (q * Vector3.UP).y
	hit_obstacle = obs != null and obs.hit(pos, body_radius())
	crashed = up_y < 0.55 or pos.y < rest_y * 0.45 or is_nan(pos.x) or is_nan(omega.x) \
		or absf(pos.x) > 25.0 or absf(pos.z) > 25.0 or hit_obstacle

func sense(target: Vector3, obs: Obstacles) -> PackedFloat32Array:
	var inv := q.inverse()
	var rel := inv * (Vector3(target.x, pos.y, target.z) - pos)
	var v := inv * vel
	var up := inv * Vector3.UP
	var s := PackedFloat32Array([rel.x / 3.0, rel.y / 3.0, rel.z / 3.0, v.x / 1.5, v.y / 1.5, v.z / 1.5,
		up.x, up.y, up.z, omega.x / 3.0, omega.y / 3.0, omega.z / 3.0])
	for i in legs:
		s.append(phi[i] / cfg.swing)
		s.append((psi[i] * side[i] - cfg.rest_splay()) / cfg.splay)
		s.append((ell[i] - cfg.leg_length) / cfg.leg_range)
		s.append(contact[i])
	for i in 12 + 4 * legs:
		s[i] = clampf(s[i] + rng.randfn(0.0, cfg.sensor_noise), -1.0, 1.0)
	s.append_array(ray_sensors(obs))
	return s

func _dist(target: Vector3) -> float:
	return Vector2(target.x - pos.x, target.z - pos.z).length()

func reward(target: Vector3) -> float:
	var d := _dist(target)
	if prev_dist < 0.0:
		prev_dist = d
	var progress := (prev_dist - d) / CTRL_DT
	prev_dist = d
	return 0.25 * maxf((q * Vector3.UP).y, 0.0) + 0.75 * clampf(progress / 0.6, -0.5, 1.0)

func retarget() -> void:
	prev_dist = -1.0

func reached(target: Vector3) -> bool:
	return _dist(target) < 0.3

func make_target(r: RandomNumberGenerator, d: float, radius: float, obs: Obstacles) -> Vector3:
	if d <= 0.0:
		return Vector3(0.0, 0.0, -1.6)
	var t := Vector3.ZERO
	for _i in 30:
		var ang := r.randf() * TAU
		var dist := lerpf(1.3, maxf(1.6, radius), d) * r.randf_range(0.7, 1.0)
		t = Vector3(sin(ang) * dist, 0.0, -cos(ang) * dist)
		if obs == null or not obs.hit(Vector3(t.x, 0.5, t.z), 0.4):
			break
	return t
