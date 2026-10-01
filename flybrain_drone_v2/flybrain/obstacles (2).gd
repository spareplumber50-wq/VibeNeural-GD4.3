class_name Obstacles
extends RefCounted
## Static obstacles: axis-aligned boxes and spheres. Used for collisions and for the
## range-finder "whisker" sensors, so the brain can learn to steer around them.

var boxes := PackedFloat32Array()      # cx, cy, cz, hx, hy, hz   (half extents)
var spheres := PackedFloat32Array()    # cx, cy, cz, radius

func count() -> int:
	return boxes.size() / 6 + spheres.size() / 4

func clear() -> void:
	boxes.clear()
	spheres.clear()

func add_box(c: Vector3, h: Vector3) -> void:
	boxes.append_array(PackedFloat32Array([c.x, c.y, c.z, h.x, h.y, h.z]))

func add_sphere(c: Vector3, r: float) -> void:
	spheres.append_array(PackedFloat32Array([c.x, c.y, c.z, r]))

func copy() -> Obstacles:
	var o := Obstacles.new()
	o.boxes = boxes.duplicate()
	o.spheres = spheres.duplicate()
	return o

func hit(p: Vector3, radius: float) -> bool:
	for i in range(0, boxes.size(), 6):
		var dx := maxf(absf(p.x - boxes[i]) - boxes[i + 3], 0.0)
		var dy := maxf(absf(p.y - boxes[i + 1]) - boxes[i + 4], 0.0)
		var dz := maxf(absf(p.z - boxes[i + 2]) - boxes[i + 5], 0.0)
		if dx * dx + dy * dy + dz * dz < radius * radius:
			return true
	for i in range(0, spheres.size(), 4):
		var d := Vector3(spheres[i], spheres[i + 1], spheres[i + 2]).distance_to(p)
		if d < spheres[i + 3] + radius:
			return true
	return false

## distance along ray (unit dir) to the nearest obstacle, or max_t
func ray(o: Vector3, d: Vector3, max_t: float) -> float:
	var best := max_t
	for i in range(0, boxes.size(), 6):
		var t0 := 0.0
		var t1 := best
		var ok := true
		for ax in 3:
			var c := boxes[i + ax]
			var h := boxes[i + 3 + ax]
			var oo := o[ax]
			var dd := d[ax]
			if absf(dd) < 1e-8:
				if oo < c - h or oo > c + h:
					ok = false
					break
			else:
				var ta := (c - h - oo) / dd
				var tb := (c + h - oo) / dd
				t0 = maxf(t0, minf(ta, tb))
				t1 = minf(t1, maxf(ta, tb))
				if t0 > t1:
					ok = false
					break
		if ok:
			best = minf(best, t0)
	for i in range(0, spheres.size(), 4):
		var oc := o - Vector3(spheres[i], spheres[i + 1], spheres[i + 2])
		var r := spheres[i + 3]
		var b := oc.dot(d)
		var disc := b * b - (oc.dot(oc) - r * r)
		if disc >= 0.0:
			var t := -b - sqrt(disc)
			if t >= 0.0:
				best = minf(best, t)
	return best

## scatter random pillars (and floating spheres for flyers) away from the given points
func random_fill(rng: RandomNumberGenerator, n: int, keep_clear: Array, ground_only: bool, area := 5.0) -> void:
	var placed := 0
	var tries := 0
	while placed < n and tries < 200:
		tries += 1
		var x := rng.randf_range(-area, area)
		var z := rng.randf_range(-area, area)
		var ok := true
		for k in keep_clear:
			if Vector2(x - k.x, z - k.z).length() < 1.3:
				ok = false
		if not ok:
			continue
		if ground_only or rng.randf() < 0.6:
			var w := rng.randf_range(0.12, 0.3)
			var hh := rng.randf_range(0.6, 1.6) if ground_only else rng.randf_range(1.0, 2.2)
			add_box(Vector3(x, hh, z), Vector3(w, hh, w))
		else:
			add_sphere(Vector3(x, rng.randf_range(0.9, 2.4), z), rng.randf_range(0.25, 0.5))
		placed += 1

func to_list() -> Array:
	var l := []
	for i in range(0, boxes.size(), 6):
		l.append(["box", boxes[i], boxes[i + 1], boxes[i + 2], boxes[i + 3], boxes[i + 4], boxes[i + 5]])
	for i in range(0, spheres.size(), 4):
		l.append(["sphere", spheres[i], spheres[i + 1], spheres[i + 2], spheres[i + 3]])
	return l

static func from_list(l: Array) -> Obstacles:
	var o := Obstacles.new()
	for it in l:
		if str(it[0]) == "box":
			o.add_box(Vector3(it[1], it[2], it[3]), Vector3(it[4], it[5], it[6]))
		else:
			o.add_sphere(Vector3(it[1], it[2], it[3]), float(it[4]))
	return o
