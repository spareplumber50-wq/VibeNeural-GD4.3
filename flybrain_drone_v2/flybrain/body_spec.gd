class_name BodySpec
extends RefCounted
## Which creature, with which settings.

var kind := "drone"            # drone | quadruped | biped
var drone := DroneConfig.new()
var walker := WalkerConfig.new()

func make(seed_: int) -> Body:
	if kind == "drone":
		return DroneBody.new(drone, seed_)
	walker.kind = kind
	return WalkerBody.new(walker, seed_)

func n_sensors() -> int:
	return make(0).n_sensors()

func n_outputs() -> int:
	return make(0).n_outputs()

func ground_only() -> bool:
	return kind != "drone"

func to_dict() -> Dictionary:
	return {"kind": kind, "drone": drone.to_dict(), "walker": walker.to_dict()}

static func from_dict(d: Dictionary) -> BodySpec:
	var s := BodySpec.new()
	s.kind = str(d.get("kind", "drone"))
	if d.has("drone"):
		s.drone = DroneConfig.from_dict(d["drone"])
	if d.has("walker"):
		s.walker = WalkerConfig.from_dict(d["walker"])
	s.walker.kind = s.kind if s.kind != "drone" else s.walker.kind
	return s

func duplicate_spec() -> BodySpec:
	return BodySpec.from_dict(to_dict())
