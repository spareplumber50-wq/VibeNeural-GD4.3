class_name DroneConfig
extends RefCounted
## Every knob of the drone. Units: kg, m, N, s.

const KEYS := ["max_tilt", "body_mass", "motor_mass", "arm_length", "max_thrust", "motor_tau",
	"drag", "yaw_coeff", "ang_damping", "wind", "gust", "sensor_noise"]

var motor_count := 4          # 4, 6 or 8 rotors (alternating spin direction)
var assist := true            # true: onboard flight controller (brain sends tilt/yaw/climb sticks)
                              # false: brain drives every motor directly (hard mode)
var max_tilt := 0.45          # rad, stick limit in assist mode
var body_mass := 0.45         # frame + battery + electronics
var motor_mass := 0.04        # per motor+prop
var arm_length := 0.16        # centre -> rotor
var max_thrust := 4.0         # per motor
var motor_tau := 0.03         # motor spin-up lag
var drag := 0.12              # linear drag N per m/s
var yaw_coeff := 0.02         # reaction torque per thrust
var ang_damping := 0.004
var wind := 0.0               # steady wind m/s (random direction per flight)
var gust := 0.0               # turbulence m/s
var sensor_noise := 0.01

func channels() -> int:
	return 4 if assist else motor_count

func total_mass() -> float:
	return body_mass + motor_mass * motor_count

func thrust_to_weight() -> float:
	return motor_count * max_thrust / (total_mass() * 9.81)

func hover_fraction() -> float:
	return 1.0 / thrust_to_weight()

func to_dict() -> Dictionary:
	var d := {"motor_count": motor_count, "assist": assist}
	for k in KEYS:
		d[k] = get(k)
	return d

static func from_dict(d: Dictionary) -> DroneConfig:
	var c := DroneConfig.new()
	if d.has("motor_count"):
		c.motor_count = int(d["motor_count"])
	if d.has("assist"):
		c.assist = bool(d["assist"])
	for k in KEYS:
		if d.has(k):
			c.set(k, float(d[k]))
	return c

func duplicate_config() -> DroneConfig:
	return DroneConfig.from_dict(to_dict())
