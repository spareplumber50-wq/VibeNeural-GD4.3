extends Node3D
## FlyBrain: train a fruit-fly connectome to fly drones and walk quadrupeds / bipeds.

const CTRL_DT := Body.CTRL_DT
const POLICY_PATH := "user://fly_pilot.json"
const GRID_SHADER := """
shader_type spatial;
void fragment() {
	vec2 uv = UV * 60.0;
	vec2 g = abs(fract(uv - 0.5) - 0.5) / fwidth(uv);
	float line = 1.0 - min(min(g.x, g.y), 1.0);
	ALBEDO = mix(vec3(0.17, 0.19, 0.21), vec3(0.42, 0.47, 0.52), line);
	ROUGHNESS = 0.9;
}
"""
const TOOLS := ["Orbit only", "Add pillar", "Add wall (across)", "Add wall (along)", "Add sphere", "Remove obstacle", "Set target"]

var spec := BodySpec.new()
var brain: FlyBrain
var brain_desc := {}
var trainer: Trainer
var runner: Runner
var genome := PackedFloat32Array()
var difficulty := 0.0
var rng := RandomNumberGenerator.new()
var hist_center: Array[float] = []
var hist_mean: Array[float] = []
var obstacles := Obstacles.new()
var scene_target := Vector3(0.0, 1.5, -3.0)
var gpu_name := ""
var gpu_error := ""

var body_node := Node3D.new()
var obstacle_root := Node3D.new()
var rotors: Array[Node3D] = []
var rotor_dirs: Array[float] = []
var leg_meshes: Array[MeshInstance3D] = []
var foot_nodes: Array[Node3D] = []
var foot_mats: Array[StandardMaterial3D] = []
var torso_node: Node3D
var target_node: MeshInstance3D
var trail_mesh := ImmediateMesh.new()
var trail: Array[Vector3] = []
var cam: Camera3D
var cam_yaw := 0.7
var cam_pitch := 0.3
var cam_dist := 7.0

var ui := {}
var stats: Label
var status: Label
var plot: Plot
var brain_view: BrainView
var train_btn: Button
var file_dialog: FileDialog
var drone_box: VBoxContainer
var walker_box: VBoxContainer
var live_acc := 0.0
var _redraw_t := 0.0

# ================================================================ setup
func _ready() -> void:
	rng.randomize()
	_detect_gpu()
	_build_world()
	_build_ui()
	_load_example(0)

func _exit_tree() -> void:
	if trainer != null:
		trainer.request_stop()
		trainer.join()
		trainer.release()
	if runner != null:
		runner.release()

func _detect_gpu() -> void:
	var rd := RenderingServer.create_local_rendering_device()
	if rd == null:
		gpu_error = "no compute device (needs Forward+ / Mobile renderer)"
		return
	gpu_name = rd.get_device_name()
	rd.free()

func _build_world() -> void:
	var env := Environment.new()
	env.background_mode = Environment.BG_SKY
	var sky := Sky.new()
	sky.sky_material = ProceduralSkyMaterial.new()
	env.sky = sky
	env.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
	env.tonemap_mode = Environment.TONE_MAPPER_FILMIC
	var we := WorldEnvironment.new()
	we.environment = env
	add_child(we)
	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-50, 30, 0)
	sun.shadow_enabled = true
	add_child(sun)
	var ground := MeshInstance3D.new()
	var pm := PlaneMesh.new()
	pm.size = Vector2(60, 60)
	ground.mesh = pm
	var sh := Shader.new()
	sh.code = GRID_SHADER
	var gm := ShaderMaterial.new()
	gm.shader = sh
	ground.material_override = gm
	add_child(ground)
	cam = Camera3D.new()
	add_child(cam)
	cam.current = true
	add_child(body_node)
	add_child(obstacle_root)
	target_node = MeshInstance3D.new()
	var sm := SphereMesh.new()
	sm.radius = 0.12
	sm.height = 0.24
	target_node.mesh = sm
	var tm := StandardMaterial3D.new()
	tm.albedo_color = Color(1.0, 0.85, 0.1)
	tm.emission_enabled = true
	tm.emission = Color(1.0, 0.7, 0.0)
	target_node.material_override = tm
	add_child(target_node)
	var trail_node := MeshInstance3D.new()
	trail_node.mesh = trail_mesh
	var trm := StandardMaterial3D.new()
	trm.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	trm.albedo_color = Color(0.3, 0.9, 1.0)
	trail_node.material_override = trm
	add_child(trail_node)

func _mat(c: Color, alpha := 1.0) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = Color(c, alpha)
	if alpha < 1.0:
		m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	return m

func _box(size: Vector3, pos: Vector3, col: Color) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	var bm := BoxMesh.new()
	bm.size = size
	mi.mesh = bm
	mi.position = pos
	mi.material_override = _mat(col)
	return mi

# ================================================================ body visuals
func _build_body_visual() -> void:
	for c in body_node.get_children():
		body_node.remove_child(c)
		c.queue_free()
	torso_node = null
	rotors.clear()
	rotor_dirs.clear()
	leg_meshes.clear()
	foot_nodes.clear()
	foot_mats.clear()
	if spec.kind == "drone":
		var cfg := spec.drone
		var model := Node3D.new()
		model.scale = Vector3.ONE * DroneBody.VIS
		body_node.add_child(model)
		model.add_child(_box(Vector3(0.1, 0.045, 0.1), Vector3.ZERO, Color(0.15, 0.16, 0.18)))
		model.add_child(_box(Vector3(0.03, 0.02, 0.05), Vector3(0, 0.02, -0.07), Color(0.9, 0.15, 0.15)))
		var rr := maxf(0.04, cfg.arm_length * 0.45)
		for i in cfg.motor_count:
			var a := TAU * (i + 0.5) / cfg.motor_count
			var p := Vector3(cos(a), 0.0, sin(a)) * cfg.arm_length
			var arm := _box(Vector3(cfg.arm_length, 0.012, 0.012), p * 0.5, Color(0.25, 0.27, 0.3))
			arm.rotation.y = -a
			model.add_child(arm)
			var rotor := Node3D.new()
			rotor.position = p + Vector3(0, 0.025, 0)
			var disc := MeshInstance3D.new()
			var cm := CylinderMesh.new()
			cm.top_radius = rr
			cm.bottom_radius = rr
			cm.height = 0.004
			disc.mesh = cm
			var dir := 1.0 if i % 2 == 0 else -1.0
			disc.material_override = _mat(Color(0.25, 0.6, 1.0) if dir > 0 else Color(1.0, 0.6, 0.2), 0.35)
			rotor.add_child(disc)
			rotor.add_child(_box(Vector3(rr * 2.0, 0.005, 0.012), Vector3.ZERO, Color(0.9, 0.9, 0.9)))
			model.add_child(rotor)
			rotors.append(rotor)
			rotor_dirs.append(dir)
	else:
		var w := spec.walker
		var torso := Node3D.new()
		torso_node = torso
		body_node.add_child(torso)
		torso.add_child(_box(Vector3(w.body_width, w.body_height, w.body_length), Vector3.ZERO, Color(0.2, 0.3, 0.45)))
		torso.add_child(_box(Vector3(w.body_width * 0.5, w.body_height * 0.6, 0.04), Vector3(0, w.body_height * 0.1, -w.body_length * 0.5), Color(0.9, 0.2, 0.2)))
		var fh := w.foot_half()
		for i in w.legs():
			var leg := MeshInstance3D.new()
			var cm := CylinderMesh.new()
			cm.top_radius = 0.013
			cm.bottom_radius = 0.013
			cm.height = 1.0
			leg.mesh = cm
			leg.material_override = _mat(Color(0.75, 0.75, 0.8))
			body_node.add_child(leg)
			leg_meshes.append(leg)
			var foot := Node3D.new()
			var fm := MeshInstance3D.new()
			if fh == Vector2.ZERO:
				var sp := SphereMesh.new()
				sp.radius = 0.03
				sp.height = 0.06
				fm.mesh = sp
			else:
				var bm := BoxMesh.new()
				bm.size = Vector3(fh.x * 2.0, 0.015, fh.y * 2.0)
				fm.mesh = bm
			var mt := _mat(Color(0.3, 0.3, 0.3))
			fm.material_override = mt
			foot.add_child(fm)
			body_node.add_child(foot)
			foot_nodes.append(foot)
			foot_mats.append(mt)

func _update_body_visual(delta: float) -> void:
	if runner == null:
		return
	var b := runner.body
	if b is DroneBody:
		var d := b as DroneBody
		if rotors.is_empty() and spec.kind != "drone":
			return
		body_node.position = d.sim.pos
		body_node.basis = Basis(d.sim.q)
		for i in mini(rotors.size(), d.sim.thrust.size()):
			rotors[i].rotate_y(rotor_dirs[i] * d.sim.thrust[i] / spec.drone.max_thrust * 60.0 * delta)
	elif b is WalkerBody and torso_node != null and leg_meshes.size() == (b as WalkerBody).legs:
		var wb := b as WalkerBody
		body_node.position = Vector3.ZERO
		body_node.basis = Basis.IDENTITY
		var bs := Basis(wb.q)
		torso_node.position = wb.pos
		torso_node.basis = bs
		for i in wb.legs:
			var a := wb.pos + bs * wb.hip[i]
			var f := wb.pos + bs * wb.foot_b[i]
			var len := maxf(a.distance_to(f), 0.001)
			var leg := leg_meshes[i]
			leg.transform = Transform3D(Basis(Quaternion(Vector3.UP, (f - a) / len)).scaled(Vector3(1, len, 1)), (a + f) * 0.5)
			foot_nodes[i].position = f
			foot_nodes[i].basis = bs
			foot_mats[i].albedo_color = Color(0.3, 0.9, 0.4) if wb.contact[i] > 0.5 else Color(0.35, 0.35, 0.35)

# ================================================================ obstacles
func _rebuild_obstacle_visuals() -> void:
	for c in obstacle_root.get_children():
		c.queue_free()
	var o := obstacles
	for i in range(0, o.boxes.size(), 6):
		obstacle_root.add_child(_box(Vector3(o.boxes[i + 3], o.boxes[i + 4], o.boxes[i + 5]) * 2.0, Vector3(o.boxes[i], o.boxes[i + 1], o.boxes[i + 2]), Color(0.75, 0.45, 0.25)))
	for i in range(0, o.spheres.size(), 4):
		var mi := MeshInstance3D.new()
		var sm := SphereMesh.new()
		sm.radius = o.spheres[i + 3]
		sm.height = o.spheres[i + 3] * 2.0
		mi.mesh = sm
		mi.position = Vector3(o.spheres[i], o.spheres[i + 1], o.spheres[i + 2])
		mi.material_override = _mat(Color(0.75, 0.3, 0.5))
		obstacle_root.add_child(mi)

func _ground_point(mouse: Vector2) -> Variant:
	var ro := cam.project_ray_origin(mouse)
	var rd := cam.project_ray_normal(mouse)
	if absf(rd.y) < 1e-4:
		return null
	var t := -ro.y / rd.y
	return ro + rd * t if t > 0.0 else null

func _click_tool(mouse: Vector2) -> void:
	var tool: int = ui.tool.selected
	if tool == 0:
		return
	var gp: Variant = _ground_point(mouse)
	if gp == null:
		return
	var p: Vector3 = gp
	var w: float = ui.size_w.value
	var h: float = ui.size_h.value
	match tool:
		1: obstacles.add_box(Vector3(p.x, h * 0.5, p.z), Vector3(w * 0.5, h * 0.5, w * 0.5))
		2: obstacles.add_box(Vector3(p.x, h * 0.5, p.z), Vector3(w * 1.5, h * 0.5, 0.1))
		3: obstacles.add_box(Vector3(p.x, h * 0.5, p.z), Vector3(0.1, h * 0.5, w * 1.5))
		4: obstacles.add_sphere(Vector3(p.x, maxf(w * 0.5, h * 0.5), p.z), w * 0.5)
		5:
			var l := obstacles.to_list()
			var best := -1
			var bd := 1.0
			for i in l.size():
				var d := Vector2(float(l[i][1]) - p.x, float(l[i][3]) - p.z).length()
				if d < bd:
					bd = d
					best = i
			if best >= 0:
				l.remove_at(best)
				obstacles = Obstacles.from_list(l)
		6:
			scene_target = Vector3(p.x, 0.0 if spec.kind != "drone" else clampf(h, 0.6, 4.0), p.z)
			ui.fixed.button_pressed = true
	_rebuild_obstacle_visuals()
	_reset_live()
	_status("Scene edited. Stop and restart training to train on the new layout.")

# ================================================================ UI
func _header(parent: Control, text: String) -> void:
	var l := Label.new()
	l.text = text
	l.add_theme_color_override("font_color", Color(0.5, 0.85, 1.0))
	l.add_theme_font_size_override("font_size", 15)
	parent.add_child(HSeparator.new())
	parent.add_child(l)

func _spin(parent: Control, key: String, label: String, lo: float, hi: float, step: float, val: float, cb: Callable = Callable()) -> void:
	var row := HBoxContainer.new()
	var l := Label.new()
	l.text = label
	l.custom_minimum_size.x = 170
	var s := SpinBox.new()
	s.min_value = lo
	s.max_value = hi
	s.step = step
	s.value = val
	s.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	if cb.is_valid():
		s.value_changed.connect(cb)
	row.add_child(l)
	row.add_child(s)
	parent.add_child(row)
	ui[key] = s

func _check(parent: Control, key: String, text: String, val: bool, cb: Callable = Callable()) -> void:
	var c := CheckBox.new()
	c.text = text
	c.button_pressed = val
	if cb.is_valid():
		c.toggled.connect(cb)
	parent.add_child(c)
	ui[key] = c

func _btn(parent: Control, text: String, cb: Callable) -> Button:
	var b := Button.new()
	b.text = text
	b.pressed.connect(cb)
	parent.add_child(b)
	return b

func _option(parent: Control, key: String, label: String, items: Array, cb: Callable) -> void:
	var row := HBoxContainer.new()
	var l := Label.new()
	l.text = label
	l.custom_minimum_size.x = 170
	var ob := OptionButton.new()
	for i in items.size():
		ob.add_item(str(items[i]), i)
	ob.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	if cb.is_valid():
		ob.item_selected.connect(cb)
	row.add_child(l)
	row.add_child(ob)
	parent.add_child(row)
	ui[key] = ob

func _build_ui() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)
	var panel := PanelContainer.new()
	panel.anchor_bottom = 1.0
	panel.offset_right = 395
	layer.add_child(panel)
	var scroll := ScrollContainer.new()
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	panel.add_child(scroll)
	var box := VBoxContainer.new()
	box.custom_minimum_size.x = 370
	scroll.add_child(box)
	var title := Label.new()
	title.text = "FlyBrain: drones and walkers"
	title.add_theme_font_size_override("font_size", 20)
	box.add_child(title)

	_header(box, "SCENE  (examples, obstacles)")
	var names: Array = []
	for e in Examples.list():
		names.append(e.name)
	_option(box, "example", "Example", names, Callable())
	_btn(box, "Load example", func() -> void: _load_example(ui.example.selected))
	_option(box, "tool", "Mouse tool (left click)", TOOLS, Callable())
	_spin(box, "size_w", "Obstacle width (m)", 0.1, 2.0, 0.05, 0.4)
	_spin(box, "size_h", "Height / target height (m)", 0.2, 4.0, 0.1, 2.0)
	_btn(box, "Clear obstacles", func() -> void:
		obstacles.clear()
		_rebuild_obstacle_visuals()
		_reset_live())
	_spin(box, "random", "Random obstacles in training", 0, 12, 1, 0)
	_check(box, "fixed", "Train on the placed target (else random)", false)

	_header(box, "BODY")
	_option(box, "kind", "Creature", ["Drone", "Quadruped", "Biped"], _on_kind_changed)
	drone_box = VBoxContainer.new()
	box.add_child(drone_box)
	_option(drone_box, "motors", "Rotors", ["4 - quadcopter", "6 - hexacopter", "8 - octocopter"], _on_body_changed)
	_check(drone_box, "assist", "Flight-controller assist (easy mode)", true, _on_body_changed)
	_spin(drone_box, "max_tilt", "Max tilt (deg)", 10, 60, 1, 26, _on_body_changed)
	_spin(drone_box, "body_mass", "Body mass (g)", 100, 3000, 10, 450, _on_body_changed)
	_spin(drone_box, "motor_mass", "Motor + prop mass (g)", 5, 200, 1, 40, _on_body_changed)
	_spin(drone_box, "arm", "Arm length (cm)", 5, 40, 0.5, 16, _on_body_changed)
	_spin(drone_box, "thrust", "Max thrust / motor (N)", 1, 20, 0.1, 4, _on_body_changed)
	_spin(drone_box, "tau", "Motor lag (ms)", 5, 150, 1, 30, _on_body_changed)
	_spin(drone_box, "drag", "Air drag (N per m/s)", 0, 0.6, 0.01, 0.12, _on_body_changed)
	_spin(drone_box, "wind", "Steady wind (m/s)", 0, 8, 0.1, 0, _on_body_changed)
	_spin(drone_box, "gust", "Gusts (m/s)", 0, 4, 0.1, 0, _on_body_changed)
	_spin(drone_box, "noise", "Sensor noise", 0, 0.1, 0.005, 0.01, _on_body_changed)
	var twr := Label.new()
	drone_box.add_child(twr)
	ui["twr"] = twr
	walker_box = VBoxContainer.new()
	box.add_child(walker_box)
	_spin(walker_box, "w_mass", "Body mass (kg)", 1, 40, 0.5, 6, _on_body_changed)
	_spin(walker_box, "w_len", "Body length (cm)", 8, 80, 1, 45, _on_body_changed)
	_spin(walker_box, "w_wid", "Hip spacing (cm)", 8, 60, 1, 26, _on_body_changed)
	_spin(walker_box, "w_hgt", "Body height (cm)", 4, 50, 1, 10, _on_body_changed)
	_spin(walker_box, "w_leg", "Leg length (cm)", 10, 70, 1, 26, _on_body_changed)
	_spin(walker_box, "w_stroke", "Leg stroke (cm)", 2, 20, 0.5, 7, _on_body_changed)
	_spin(walker_box, "w_swing", "Hip swing (deg)", 10, 60, 1, 34, _on_body_changed)
	_spin(walker_box, "w_fric", "Ground friction", 0.2, 2.0, 0.05, 1.0, _on_body_changed)
	_spin(walker_box, "w_tau", "Joint lag (ms)", 10, 200, 5, 50, _on_body_changed)

	_header(box, "FLY BRAIN  (the connectome)")
	var bi := Label.new()
	bi.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	box.add_child(bi)
	ui["brain_info"] = bi
	_spin(box, "neurons", "Neurons (synthetic)", 60, 30000, 10, 300)
	_spin(box, "sens", "Sensory neurons", 8, 256, 1, 32)
	_spin(box, "outs", "Descending neurons", 8, 256, 1, 32)
	_spin(box, "deg", "Synapses in / neuron", 4, 60, 1, 12)
	_spin(box, "seed", "Wiring seed", 0, 99999, 1, 1)
	_btn(box, "Generate synthetic connectome", func() -> void: _make_brain(_desc_from_ui()))
	_btn(box, "Import real connectome (CSV edge list)...", func() -> void: file_dialog.popup_centered(Vector2i(820, 520)))
	file_dialog = FileDialog.new()
	file_dialog.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	file_dialog.access = FileDialog.ACCESS_FILESYSTEM
	file_dialog.filters = PackedStringArray(["*.csv,*.tsv,*.txt ; Edge list (pre_id, post_id, weight)"])
	file_dialog.file_selected.connect(_on_csv_selected)
	add_child(file_dialog)

	_header(box, "COMPUTE")
	var gl := Label.new()
	gl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	gl.text = ("GPU compute device: %s" % gpu_name) if gpu_name != "" else ("CPU threads only: %s" % gpu_error)
	box.add_child(gl)
	_check(box, "use_gpu", "Run the brain on the GPU (training)", gpu_name != "")
	if gpu_name == "":
		ui.use_gpu.disabled = true

	_header(box, "TRAINING  (evolution strategy)")
	_spin(box, "pop", "Population", 8, 1024, 2, 64)
	_spin(box, "sigma", "Mutation size (sigma)", 0.005, 0.3, 0.005, 0.03)
	_spin(box, "lr", "Learning rate", 0.001, 0.2, 0.001, 0.02)
	_spin(box, "episodes", "Flights per candidate", 1, 8, 1, 2)
	_spin(box, "target_range", "Max target distance (m)", 0, 6, 0.25, 2)
	_spin(box, "duration", "Flight length (s)", 3, 30, 1, 8)
	train_btn = _btn(box, "Start training", _toggle_training)
	_btn(box, "Reset policy (untrained brain)", _reset_policy)

	_header(box, "RUN")
	_check(box, "moving", "Moving target", true, func(_v) -> void: _reset_live())
	_btn(box, "New target", func() -> void: if runner != null: runner.new_target())
	_btn(box, "Respawn", _reset_live)
	var hb := HBoxContainer.new()
	_btn(hb, "Save policy", _save_policy)
	_btn(hb, "Load policy", _load_policy)
	box.add_child(hb)
	status = Label.new()
	status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	box.add_child(status)

	var right := VBoxContainer.new()
	right.anchor_left = 1.0
	right.anchor_right = 1.0
	right.offset_left = -340
	right.offset_right = -10
	right.offset_top = 10
	layer.add_child(right)
	stats = Label.new()
	right.add_child(stats)
	plot = Plot.new()
	plot.custom_minimum_size = Vector2(330, 130)
	right.add_child(plot)
	brain_view = BrainView.new()
	brain_view.custom_minimum_size = Vector2(330, 300)
	right.add_child(brain_view)
	var help := Label.new()
	help.text = "Brain view: orange = firing, blue = inhibited\nsensory neurons left, descending right\nRight-drag = orbit, wheel = zoom\n(left-drag also orbits when the tool is 'Orbit only')"
	help.add_theme_font_size_override("font_size", 11)
	right.add_child(help)

func _status(msg: String) -> void:
	status.text = msg

# ================================================================ spec <-> UI
func _read_spec() -> void:
	spec.kind = ["drone", "quadruped", "biped"][ui.kind.selected]
	var d := spec.drone
	d.motor_count = [4, 6, 8][ui.motors.selected]
	d.assist = ui.assist.button_pressed
	d.max_tilt = deg_to_rad(ui.max_tilt.value)
	d.body_mass = ui.body_mass.value / 1000.0
	d.motor_mass = ui.motor_mass.value / 1000.0
	d.arm_length = ui.arm.value / 100.0
	d.max_thrust = ui.thrust.value
	d.motor_tau = ui.tau.value / 1000.0
	d.drag = ui.drag.value
	d.wind = ui.wind.value
	d.gust = ui.gust.value
	d.sensor_noise = ui.noise.value
	var w := spec.walker
	w.kind = spec.kind if spec.kind != "drone" else w.kind
	w.body_mass = ui.w_mass.value
	w.body_length = ui.w_len.value / 100.0
	w.body_width = ui.w_wid.value / 100.0
	w.body_height = ui.w_hgt.value / 100.0
	w.leg_length = ui.w_leg.value / 100.0
	w.leg_range = ui.w_stroke.value / 100.0
	w.swing = deg_to_rad(ui.w_swing.value)
	w.friction = ui.w_fric.value
	w.motor_tau = ui.w_tau.value / 1000.0

func _spec_to_ui() -> void:
	ui.kind.select(["drone", "quadruped", "biped"].find(spec.kind))
	var d := spec.drone
	ui.motors.select([4, 6, 8].find(d.motor_count))
	ui.assist.set_pressed_no_signal(d.assist)
	ui.max_tilt.set_value_no_signal(rad_to_deg(d.max_tilt))
	ui.body_mass.set_value_no_signal(d.body_mass * 1000.0)
	ui.motor_mass.set_value_no_signal(d.motor_mass * 1000.0)
	ui.arm.set_value_no_signal(d.arm_length * 100.0)
	ui.thrust.set_value_no_signal(d.max_thrust)
	ui.tau.set_value_no_signal(d.motor_tau * 1000.0)
	ui.drag.set_value_no_signal(d.drag)
	ui.wind.set_value_no_signal(d.wind)
	ui.gust.set_value_no_signal(d.gust)
	ui.noise.set_value_no_signal(d.sensor_noise)
	var w := spec.walker
	ui.w_mass.set_value_no_signal(w.body_mass)
	ui.w_len.set_value_no_signal(w.body_length * 100.0)
	ui.w_wid.set_value_no_signal(w.body_width * 100.0)
	ui.w_hgt.set_value_no_signal(w.body_height * 100.0)
	ui.w_leg.set_value_no_signal(w.leg_length * 100.0)
	ui.w_stroke.set_value_no_signal(w.leg_range * 100.0)
	ui.w_swing.set_value_no_signal(rad_to_deg(w.swing))
	ui.w_fric.set_value_no_signal(w.friction)
	ui.w_tau.set_value_no_signal(w.motor_tau * 1000.0)
	drone_box.visible = spec.kind == "drone"
	walker_box.visible = spec.kind != "drone"
	_update_twr()

func _update_twr() -> void:
	var d := spec.drone
	var warn := "" if d.thrust_to_weight() >= 1.3 else "   (too heavy to fly well!)"
	ui.twr.text = "Thrust/weight %.2f   hover throttle %.0f %%%s" % [d.thrust_to_weight(), d.hover_fraction() * 100.0, warn]

func _on_kind_changed(_i: int) -> void:
	var k: String = ["drone", "quadruped", "biped"][ui.kind.selected]
	if k != "drone":
		spec.walker = WalkerConfig.defaults(k)
	spec.kind = k
	_spec_to_ui()
	_body_changed_apply()

func _on_body_changed(_v = null) -> void:
	if brain == null:
		return
	_read_spec()
	_update_twr()
	_body_changed_apply()

func _body_changed_apply() -> void:
	_build_body_visual()
	if genome.size() != BrainBatch.genome_size(brain, spec.n_sensors(), spec.n_outputs()):
		_reset_policy()
		_status("Body changed: new policy (sensor / motor layout differs).")
	else:
		_reset_live()
		if trainer != null and trainer.is_running():
			_status("Body edited. Stop and restart training to train on the new body.")

func _desc_from_ui() -> Dictionary:
	var sens := int(ui.sens.value)
	var outs := int(ui.outs.value)
	return {"type": "synthetic", "n": maxi(int(ui.neurons.value), sens + outs + 8), "sens": sens,
		"out": outs, "deg": int(ui.deg.value), "seed": int(ui.seed.value)}

func _make_brain(desc: Dictionary) -> bool:
	var b: FlyBrain
	if desc.get("type", "synthetic") == "csv":
		b = FlyBrain.from_csv(desc.path, int(desc.sens), int(desc.out), int(desc.seed))
	else:
		b = FlyBrain.synthetic(int(desc.n), int(desc.sens), int(desc.out), int(desc.deg), int(desc.seed))
	if b == null:
		_status("Could not read that connectome. Need an edge list: pre_id,post_id,weight with enough neurons.")
		return false
	brain = b
	brain_desc = desc
	brain_view.set_brain(b)
	ui.brain_info.text = b.summary()
	_reset_policy()
	return true

func _on_csv_selected(path: String) -> void:
	var d := _desc_from_ui()
	d["type"] = "csv"
	d["path"] = path
	if _make_brain(d):
		_status("Imported %s" % path.get_file())

func _load_example(i: int) -> void:
	var ex: Dictionary = Examples.list()[i]
	ui.example.select(i)
	spec = BodySpec.new()
	spec.kind = ex.kind
	if ex.kind == "drone":
		spec.drone = DroneConfig.from_dict(ex.get("drone", {}))
	else:
		spec.walker = WalkerConfig.defaults(ex.kind)
		for k in ex.get("walker", {}):
			spec.walker.set(k, ex.walker[k])
	obstacles = Obstacles.from_list(ex.obstacles)
	var t: Array = ex.get("target", [0.0, 1.5 if ex.kind == "drone" else 0.0, -3.0])
	scene_target = Vector3(t[0], t[1], t[2])
	ui.random.value = ex.random
	ui.target_range.value = ex.range
	ui.moving.set_pressed_no_signal(ex.moving)
	ui.fixed.set_pressed_no_signal(ex.fixed)
	_spec_to_ui()
	_rebuild_obstacle_visuals()
	_build_body_visual()
	if brain == null:
		_make_brain(_desc_from_ui())
	else:
		_reset_policy()
	_status("%s\n%s" % [ex.name, ex.desc])

# ================================================================ policy / training
func _reset_policy() -> void:
	if trainer != null:
		trainer.request_stop()
		trainer.join()
		trainer.release()
		trainer = null
	train_btn.text = "Start training"
	genome = BrainBatch.init_genome(brain, spec.n_sensors(), spec.n_outputs(), rng)
	difficulty = 0.0
	hist_center.clear()
	hist_mean.clear()
	plot.a = hist_center
	plot.b = hist_mean
	plot.queue_redraw()
	stats.text = "Untrained brain"
	_reset_live()

func _toggle_training() -> void:
	if trainer != null and trainer.is_running():
		trainer.request_stop()
		train_btn.text = "Stopping..."
		return
	_read_spec()
	if trainer == null:
		trainer = Trainer.new(brain, spec.duplicate_spec(), genome)
		trainer.generation_done.connect(_on_gen.bind(trainer), CONNECT_DEFERRED)
		trainer.difficulty = difficulty
	trainer.spec = spec.duplicate_spec()
	trainer.obstacles = obstacles.copy()
	trainer.random_obstacles = int(ui.random.value)
	trainer.fixed_target = ui.fixed.button_pressed
	trainer.scene_target = scene_target
	trainer.use_gpu = ui.use_gpu.button_pressed
	trainer.pop = maxi(8, int(ui.pop.value) / 2 * 2)
	trainer.sigma = ui.sigma.value
	trainer.lr = ui.lr.value
	trainer.episodes = int(ui.episodes.value)
	trainer.max_radius = ui.target_range.value
	trainer.duration = ui.duration.value
	trainer.start()
	train_btn.text = "Stop training"
	_status("Training on background threads. The body on screen always runs the latest policy.")

func _on_gen(info: Dictionary, t: Trainer) -> void:
	if t != trainer:
		return
	genome = info.theta
	difficulty = info.difficulty
	hist_center.append(info.center)
	hist_mean.append(info.mean)
	plot.queue_redraw()
	if runner != null:
		runner.set_genome(genome)
		runner.difficulty = difficulty
	stats.text = "Generation %d  (%.1f s each)\npolicy fitness %.2f   population best %.2f\ntask difficulty %.0f %%\nbrain on: %s" % [
		info.gen, float(info.ms) / 1000.0, info.center, info.best, info.difficulty * 100.0,
		("GPU (%s)" % info.device) if info.gpu else "CPU threads"]

func _save_policy() -> void:
	var d := {"version": 2, "spec": spec.to_dict(), "brain": brain_desc, "genome": Array(genome), "difficulty": difficulty,
		"obstacles": obstacles.to_list(), "target": [scene_target.x, scene_target.y, scene_target.z]}
	var f := FileAccess.open(POLICY_PATH, FileAccess.WRITE)
	if f == null:
		_status("Could not write policy file.")
		return
	f.store_string(JSON.stringify(d))
	_status("Saved to %s" % ProjectSettings.globalize_path(POLICY_PATH))

func _load_policy() -> void:
	if not FileAccess.file_exists(POLICY_PATH):
		_status("No saved policy yet.")
		return
	var d: Variant = JSON.parse_string(FileAccess.get_file_as_string(POLICY_PATH))
	if typeof(d) != TYPE_DICTIONARY or not d.has("genome") or not d.has("brain") or not d.has("spec"):
		_status("Policy file is not valid (or from an older version).")
		return
	spec = BodySpec.from_dict(d.spec)
	obstacles = Obstacles.from_list(d.get("obstacles", []))
	var t: Array = d.get("target", [0.0, 1.5, -3.0])
	scene_target = Vector3(t[0], t[1], t[2])
	_spec_to_ui()
	_rebuild_obstacle_visuals()
	_build_body_visual()
	if not _make_brain(d.brain):
		return
	var g := PackedFloat32Array(d.genome)
	if g.size() != BrainBatch.genome_size(brain, spec.n_sensors(), spec.n_outputs()):
		_status("Policy does not match this brain / body.")
		return
	genome = g
	difficulty = float(d.get("difficulty", 0.0))
	_reset_live()
	_status("Policy loaded.")

# ================================================================ live run
func _reset_live() -> void:
	if brain == null or genome.size() != BrainBatch.genome_size(brain, spec.n_sensors(), spec.n_outputs()):
		return
	if runner != null:
		runner.release()
	runner = Runner.new(brain, spec.duplicate_spec(), genome, obstacles.copy(), difficulty, ui.use_gpu.button_pressed)
	runner.max_radius = ui.target_range.value
	runner.moving = ui.moving.button_pressed
	runner.fixed_target = ui.fixed.button_pressed
	runner.scene_target = scene_target
	runner.new_target()
	trail.clear()

func _process(delta: float) -> void:
	if trainer != null and trainer.poll():
		train_btn.text = "Start training"
	if runner != null:
		live_acc += minf(delta, 0.1)
		while live_acc >= CTRL_DT:
			live_acc -= CTRL_DT
			runner.step()
			if runner.tick % 2 == 0 and not runner.body.crashed:
				trail.append(runner.body.pos)
				if trail.size() > 250:
					trail.pop_front()
		_update_body_visual(delta)
		target_node.position = runner.target + Vector3(0, 0.12 if spec.kind != "drone" else 0.0, 0)
		if Engine.get_process_frames() % 3 == 0:
			brain_view.rates = runner.batch.rates(0)
	trail_mesh.clear_surfaces()
	if trail.size() > 1:
		trail_mesh.surface_begin(Mesh.PRIMITIVE_LINE_STRIP)
		for p in trail:
			trail_mesh.surface_add_vertex(p)
		trail_mesh.surface_end()
	var focus := Vector3(0, 1.0, -1.5)
	cam.position = focus + Vector3(cos(cam_pitch) * sin(cam_yaw), sin(cam_pitch), cos(cam_pitch) * cos(cam_yaw)) * cam_dist
	cam.look_at(focus)
	_redraw_t += delta
	if _redraw_t > 0.06:
		_redraw_t = 0.0
		brain_view.queue_redraw()

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseMotion:
		var orbit: bool = (event.button_mask & (MOUSE_BUTTON_MASK_RIGHT | MOUSE_BUTTON_MASK_MIDDLE)) != 0 \
			or ((event.button_mask & MOUSE_BUTTON_MASK_LEFT) != 0 and ui.tool.selected == 0)
		if orbit:
			cam_yaw -= event.relative.x * 0.006
			cam_pitch = clampf(cam_pitch + event.relative.y * 0.006, 0.05, 1.4)
	elif event is InputEventMouseButton and event.pressed:
		if event.button_index == MOUSE_BUTTON_WHEEL_UP:
			cam_dist = maxf(2.0, cam_dist * 0.92)
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			cam_dist = minf(30.0, cam_dist * 1.08)
		elif event.button_index == MOUSE_BUTTON_LEFT:
			_click_tool(event.position)

# ================================================================ little widgets
class Plot extends Control:
	var a: Array[float] = []
	var b: Array[float] = []
	func _draw() -> void:
		draw_rect(Rect2(Vector2.ZERO, size), Color(0.05, 0.06, 0.08))
		_line(b, Color(0.55, 0.55, 0.6))
		_line(a, Color(0.3, 1.0, 0.5))
		draw_string(ThemeDB.fallback_font, Vector2(6, 14), "fitness  (green = policy, grey = mean; crash = negative)", HORIZONTAL_ALIGNMENT_LEFT, -1, 11)
	func _line(v: Array[float], col: Color) -> void:
		if v.size() < 2:
			return
		var pts := PackedVector2Array()
		for i in v.size():
			pts.append(Vector2(size.x * i / float(v.size() - 1), size.y * (1.0 - clampf((v[i] + 0.4) / 1.4, 0.0, 1.0))))
		draw_polyline(pts, col, 1.5)

class BrainView extends Control:
	var brain: FlyBrain
	var rates := PackedFloat32Array()
	var edges := PackedVector2Array()
	func set_brain(b: FlyBrain) -> void:
		brain = b
		rates = PackedFloat32Array()
		edges.clear()
		var r := RandomNumberGenerator.new()
		r.seed = 1
		for _i in mini(350, b.pre.size()):
			var e := r.randi() % b.pre.size()
			edges.append(b.layout[b.pre[e]])
			edges.append(b.layout[b.post[e]])
	func _draw() -> void:
		draw_rect(Rect2(Vector2.ZERO, size), Color(0.05, 0.06, 0.08))
		if brain == null:
			return
		for i in range(0, edges.size(), 2):
			draw_line(edges[i] * size, edges[i + 1] * size, Color(1, 1, 1, 0.05), 1.0)
		var step := maxi(1, brain.n / 3000)        # big brains: draw a subset
		for i in range(0, brain.n, step):
			var a := rates[i] if i < rates.size() else 0.0
			var m := absf(a)
			var col := Color(1.0, 0.55, 0.1) if a >= 0.0 else Color(0.25, 0.6, 1.0)
			draw_circle(brain.layout[i] * size, 1.5 + 2.5 * m, Color(col, 0.25 + 0.75 * minf(m * 2.0, 1.0)))
