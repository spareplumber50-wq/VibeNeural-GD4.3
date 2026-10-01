class_name Runner
extends RefCounted
## Live flight / walk shown on screen: one body driven by one brain, stepped at 50 Hz.

var spec: BodySpec
var brain: FlyBrain
var body: Body
var batch: BrainBatch
var obs: Obstacles
var rng := RandomNumberGenerator.new()
var target := Vector3.ZERO
var tick := 0
var difficulty := 0.0
var max_radius := 2.0
var moving := true
var fixed_target := false
var scene_target := Vector3(0.0, 1.5, -3.0)
var crash_timer := 0.0
var genome := PackedFloat32Array()

func _init(b: FlyBrain, sp: BodySpec, g: PackedFloat32Array, o: Obstacles, d: float, want_gpu: bool) -> void:
	brain = b
	spec = sp
	obs = o
	genome = g
	difficulty = d
	rng.randomize()
	# the GPU only pays off for big networks; small ones are faster on the CPU for a single agent
	batch = BrainBatch.new(b, sp.n_sensors(), sp.n_outputs(), 1, 1, Body.CTRL_DT, want_gpu and b.pre.size() > 20000)
	batch.set_genomes(g)
	reset()

func release() -> void:
	batch.release()

func set_genome(g: PackedFloat32Array) -> void:
	genome = g
	batch.set_genomes(g)

func reset() -> void:
	body = spec.make(rng.randi())
	body.reset(rng, difficulty, obs)
	batch.reset_state()
	tick = 0
	crash_timer = 0.0
	new_target()

func new_target() -> void:
	if fixed_target:
		target = scene_target if not spec.ground_only() else Vector3(scene_target.x, 0.0, scene_target.z)
	else:
		target = body.make_target(rng, maxf(difficulty, 0.3), max_radius, obs)
	body.retarget()

func step() -> void:
	if body.crashed:
		crash_timer += Body.CTRL_DT
		if crash_timer > 1.2:
			reset()
		return
	tick += 1
	if (moving and tick % int(body.retarget_period() / Body.CTRL_DT) == 0) or body.reached(target):
		new_target()
	body.step(batch.step(body.sense(target, obs)), obs)
