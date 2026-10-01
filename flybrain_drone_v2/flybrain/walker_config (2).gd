class_name WalkerConfig
extends RefCounted
## Legged robot: a torso carried by stiff, position-controlled legs (hip swing, hip splay,
## leg extension per leg). Quadruped = 4 point feet; biped = 2 flat feet with 4 contact points.

const KEYS := ["body_mass", "body_length", "body_width", "body_height", "leg_length", "leg_range",
	"swing", "splay", "friction", "damping_ratio", "motor_tau", "sensor_noise"]

var kind := "quadruped"
var body_mass := 6.0
var body_length := 0.45      # front-back
var body_width := 0.26       # hip spacing
var body_height := 0.10
var leg_length := 0.26
var leg_range := 0.07        # leg stroke (+/-)
var swing := 0.6             # max hip swing (rad)
var splay := 0.25            # max hip splay (rad)
var friction := 1.0
var damping_ratio := 0.8
var motor_tau := 0.05        # joint lag
var sensor_noise := 0.01

func legs() -> int:
	return 4 if kind == "quadruped" else 2

func foot_half() -> Vector2:          # (half width, half length) of the foot; 0 = point foot
	return Vector2.ZERO if kind == "quadruped" else Vector2(0.035, 0.07)

func rest_splay() -> float:
	return 0.12 if kind == "quadruped" else 0.0

func hips() -> Array[Vector3]:
	var h: Array[Vector3] = []
	var y := -body_height * 0.5
	if kind == "quadruped":
		for zs in [-1.0, 1.0]:
			for xs in [1.0, -1.0]:
				h.append(Vector3(xs * body_width * 0.5, y, zs * body_length * 0.5))
	else:
		h.append(Vector3(body_width * 0.5, y, 0.0))
		h.append(Vector3(-body_width * 0.5, y, 0.0))
	return h

static func defaults(k: String) -> WalkerConfig:
	var c := WalkerConfig.new()
	c.kind = k
	if k == "biped":
		c.body_mass = 5.0
		c.body_length = 0.14
		c.body_width = 0.22
		c.body_height = 0.28
		c.leg_length = 0.40
		c.leg_range = 0.08
		c.swing = 0.5
		c.splay = 0.15
		c.friction = 1.2
	return c

func to_dict() -> Dictionary:
	var d := {"kind": kind}
	for k in KEYS:
		d[k] = get(k)
	return d

static func from_dict(d: Dictionary) -> WalkerConfig:
	var c := WalkerConfig.new()
	c.kind = str(d.get("kind", "quadruped"))
	for k in KEYS:
		if d.has(k):
			c.set(k, float(d[k]))
	return c

func duplicate_config() -> WalkerConfig:
	return WalkerConfig.from_dict(to_dict())
