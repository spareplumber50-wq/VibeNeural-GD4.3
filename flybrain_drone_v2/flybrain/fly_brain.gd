class_name FlyBrain
extends RefCounted
## The "wiring diagram": a sparse, signed, directed synapse graph.
## Import a real connectome (e.g. a FlyWire subset) from CSV, or generate a
## connectome-like synthetic one so the project runs out of the box.

var n := 0
var pre := PackedInt32Array()
var post := PackedInt32Array()
var w := PackedFloat32Array()          # signed, row-normalised weights
var sensory := PackedInt32Array()
var output := PackedInt32Array()
var tau := PackedFloat32Array()        # membrane time constants (s)
var layout := PackedVector2Array()     # 0..1 positions for drawing
var role := PackedByteArray()          # 0 sensory, 1 inter, 2 descending/output
var source := "synthetic"

static func synthetic(n_total: int, n_sens: int, n_out: int, in_degree: int, seed_: int) -> FlyBrain:
	var b := FlyBrain.new()
	var rng := RandomNumberGenerator.new()
	rng.seed = seed_
	b.n = n_total
	var n_inter := maxi(1, n_total - n_sens - n_out)
	b.n = n_sens + n_inter + n_out
	b.role.resize(b.n)
	var sgn := PackedFloat32Array()
	sgn.resize(b.n)
	for i in b.n:
		b.role[i] = 0 if i < n_sens else (1 if i < n_sens + n_inter else 2)
		# Dale's law: 20 % inhibitory, scaled 4x so excitation/inhibition balance
		sgn[i] = -4.0 if (b.role[i] == 1 and rng.randf() < 0.2) else 1.0
	var module_size := 24
	for j in range(n_sens, b.n):
		var is_out := j >= n_sens + n_inter
		var mod_start := n_sens + floori(float(j - n_sens) / module_size) * module_size
		for _k in in_degree:
			var i := 0
			if is_out:
				i = n_sens + rng.randi() % n_inter
			else:
				var u := rng.randf()
				if u < 0.15 and n_sens > 0:
					i = rng.randi() % n_sens
				elif u < 0.7:
					i = mod_start + rng.randi() % mini(module_size, n_sens + n_inter - mod_start)
				else:
					i = n_sens + rng.randi() % n_inter
			if i == j:
				continue
			b.pre.append(i)
			b.post.append(j)
			b.w.append(sgn[i] * exp(rng.randfn(0.0, 0.7)))   # log-normal synapse counts
	# layout: sensory left, output right, interneuron modules clustered between
	b.layout.resize(b.n)
	var n_mod := ceili(float(n_inter) / module_size)
	var cols := ceili(sqrt(float(n_mod)))
	var rows := ceili(float(n_mod) / cols)
	for i in b.n:
		if b.role[i] == 0:
			b.layout[i] = Vector2(0.04, (i + 0.5) / n_sens)
		elif b.role[i] == 2:
			b.layout[i] = Vector2(0.96, (i - n_sens - n_inter + 0.5) / n_out)
		else:
			var m := floori(float(i - n_sens) / module_size)
			var c := Vector2(0.15 + 0.7 * ((m % cols) + 0.5) / cols, 0.05 + 0.9 * (floori(float(m) / cols) + 0.5) / rows)
			b.layout[i] = c + Vector2(rng.randfn(0.0, 0.025), rng.randfn(0.0, 0.025))
	b.finalize(seed_)
	return b

## edges CSV: pre_id,post_id,weight   (weight signed: negative = inhibitory)
## optional neurons.csv next to it: id,role   (role: sensory | inter | descending)
static func from_csv(edges_path: String, n_sens: int, n_out: int, seed_: int) -> FlyBrain:
	var f := FileAccess.open(edges_path, FileAccess.READ)
	if f == null:
		return null
	var ids := {}
	var pre_l := PackedInt32Array()
	var post_l := PackedInt32Array()
	var w_l := PackedFloat32Array()
	while not f.eof_reached():
		var line := f.get_line().strip_edges()
		if line.is_empty():
			continue
		var parts := line.split(",")
		if parts.size() < 3:
			parts = line.split("\t")
		if parts.size() < 3 or not parts[0].strip_edges().is_valid_int():
			continue
		var a := parts[0].strip_edges().to_int()
		var c := parts[1].strip_edges().to_int()
		var wt := parts[2].strip_edges().to_float()
		if wt == 0.0:
			continue
		if not ids.has(a):
			ids[a] = ids.size()
		if not ids.has(c):
			ids[c] = ids.size()
		pre_l.append(ids[a])
		post_l.append(ids[c])
		w_l.append(wt)
	f.close()
	if ids.size() < n_sens + n_out + 4:
		return null
	var b := FlyBrain.new()
	b.source = edges_path.get_file()
	b.n = ids.size()
	b.pre = pre_l
	b.post = post_l
	b.w = w_l
	var indeg := PackedFloat32Array()
	var outdeg := PackedFloat32Array()
	indeg.resize(b.n)
	outdeg.resize(b.n)
	for e in pre_l.size():
		outdeg[pre_l[e]] += 1.0
		indeg[post_l[e]] += 1.0
	b.role.resize(b.n)
	b.role.fill(1)
	# roles from neurons.csv if present, otherwise infer from degree
	var rf := FileAccess.open(edges_path.get_base_dir().path_join("neurons.csv"), FileAccess.READ)
	var have_roles := false
	if rf != null:
		while not rf.eof_reached():
			var p := rf.get_line().strip_edges().split(",")
			if p.size() >= 2 and p[0].strip_edges().is_valid_int() and ids.has(p[0].strip_edges().to_int()):
				var r := p[1].strip_edges().to_lower()
				var idx: int = ids[p[0].strip_edges().to_int()]
				if r.begins_with("s"):
					b.role[idx] = 0
					have_roles = true
				elif r.begins_with("d") or r.begins_with("o") or r.begins_with("m"):
					b.role[idx] = 2
					have_roles = true
	var order: Array = range(b.n)
	if have_roles:
		var sens: Array = order.filter(func(i): return b.role[i] == 0)
		var outs: Array = order.filter(func(i): return b.role[i] == 2)
		sens.sort_custom(func(x, y): return outdeg[x] > outdeg[y])
		outs.sort_custom(func(x, y): return indeg[x] > indeg[y])
		for k in sens.size():
			if k >= n_sens:
				b.role[sens[k]] = 1
		for k in outs.size():
			if k >= n_out:
				b.role[outs[k]] = 1
	else:
		order.sort_custom(func(x, y): return (outdeg[x] - indeg[x]) > (outdeg[y] - indeg[y]))
		for k in n_sens:
			b.role[order[k]] = 0
		for k in n_out:
			b.role[order[b.n - 1 - k]] = 2
	b.finalize(seed_)
	return b

func finalize(seed_: int) -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = seed_ + 1
	# row-normalise so recurrent gain is controllable (trained as one scalar)
	var ss := PackedFloat32Array()
	ss.resize(n)
	for e in pre.size():
		ss[post[e]] += w[e] * w[e]
	for e in pre.size():
		var s := ss[post[e]]
		if s > 0.0:
			w[e] /= sqrt(s)
	tau.resize(n)
	for i in n:
		tau[i] = exp(rng.randf_range(log(0.04), log(0.4)))
	sensory.clear()
	output.clear()
	for i in n:
		if role[i] == 0:
			sensory.append(i)
		elif role[i] == 2:
			output.append(i)
	if layout.size() != n:
		layout.resize(n)
		var si := 0
		var oi := 0
		for i in n:
			if role[i] == 0:
				layout[i] = Vector2(0.04, (si + 0.5) / maxi(1, sensory.size()))
				si += 1
			elif role[i] == 2:
				layout[i] = Vector2(0.96, (oi + 0.5) / maxi(1, output.size()))
				oi += 1
			else:
				layout[i] = Vector2(rng.randf_range(0.12, 0.88), rng.randf_range(0.03, 0.97))

func summary() -> String:
	return "%s: %d neurons, %d synapses (%d sensory in, %d motor out)" % [source, n, pre.size(), sensory.size(), output.size()]
