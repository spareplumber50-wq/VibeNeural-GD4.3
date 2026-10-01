class_name Trainer
extends RefCounted
## Evolution strategy (mirrored sampling + rank shaping + Adam). Every generation all
## candidates x flights fly in lock-step: bodies are simulated on worker threads, the fly
## brain for all of them advances in ONE GPU dispatch per tick (or on CPU threads).

signal generation_done(info: Dictionary)

class Agent:
	var body: Body
	var rng := RandomNumberGenerator.new()
	var obs: Obstacles
	var target := Vector3.ZERO
	var alive := true
	var score := 0.0
	var sens := PackedFloat32Array()

var brain: FlyBrain
var spec: BodySpec
var obstacles := Obstacles.new()     # the layout from the scene editor
var random_obstacles := 0            # extra random obstacles (scaled by difficulty)
var fixed_target := false
var scene_target := Vector3(0.0, 1.5, -3.0)
var use_gpu := true
var theta := PackedFloat32Array()
var pop := 32
var sigma := 0.03
var lr := 0.02
var episodes := 2
var max_radius := 2.0
var duration := 8.0
var difficulty := 0.0
var gen := 0
var gpu_active := false
var gpu_name := ""
var gpu_error := ""

var _m := PackedFloat32Array()
var _v := PackedFloat32Array()
var _t := 0
var _thread: Thread
var _stop := false
var _batch: BrainBatch
var _pop_active := 0
var _agents: Array = []
var _outs := PackedFloat32Array()
var _chunk := 1
var _tick := 0
var _period := 200
var _total_ticks := 1
var _S := 0
var _C := 0

func _init(b: FlyBrain, sp: BodySpec, g: PackedFloat32Array) -> void:
	brain = b
	spec = sp
	theta = g.duplicate()
	_m.resize(theta.size())
	_v.resize(theta.size())

func is_running() -> bool:
	return _thread != null
func start() -> void:
	if _thread != null:
		return
	_stop = false
	_thread = Thread.new()
	_thread.start(_loop)
func request_stop() -> void:
	_stop = true
func join() -> void:
	if _thread != null:
		_thread.wait_to_finish()
		_thread = null
## main thread: returns true once when the worker thread has finished
func poll() -> bool:
	if _thread != null and not _thread.is_alive():
		_thread.wait_to_finish()
		_thread = null
		return true
	return false

func _loop() -> void:
	while not _stop:
		var info := step_generation()
		call_deferred("emit_signal", "generation_done", info)
	release()

func release() -> void:
	if _batch != null:
		_batch.release()
		_batch = null

func _make_target(ag: Agent) -> Vector3:
	if fixed_target:
		return scene_target if not spec.ground_only() else Vector3(scene_target.x, 0.0, scene_target.z)
	return ag.body.make_target(ag.rng, difficulty, max_radius, ag.obs)

func _sense_chunk(ci: int) -> void:
	for a in range(ci * _chunk, mini((ci + 1) * _chunk, _agents.size())):
		var ag: Agent = _agents[a]
		if not ag.alive:
			continue
		if _tick % _period == 0 or ag.body.reached(ag.target):
			ag.target = _make_target(ag)
			ag.body.retarget()
		ag.sens = ag.body.sense(ag.target, ag.obs)

func _phys_chunk(ci: int) -> void:
	for a in range(ci * _chunk, mini((ci + 1) * _chunk, _agents.size())):
		var ag: Agent = _agents[a]
		if not ag.alive:
			continue
		ag.body.step(_outs.slice(a * _C, a * _C + _C), ag.obs)
		if ag.body.crashed:
			# crashing must never pay: every remaining tick of the flight is charged a penalty
			ag.alive = false
			ag.score -= ag.body.crash_penalty() * float(_total_ticks - _tick)
		else:
			ag.score += clampf(ag.body.reward(ag.target), -0.2, 1.0)

## flat = (n_genomes x G) parameter vectors -> mean fitness of each
func _evaluate(flat: PackedFloat32Array, n_gen: int) -> PackedFloat32Array:
	_S = spec.n_sensors()
	_C = spec.n_outputs()
	var A := n_gen * episodes
	if _batch == null or _pop_active != n_gen:
		release()
		_batch = BrainBatch.new(brain, _S, _C, A, episodes, Body.CTRL_DT, use_gpu)
		_pop_active = n_gen
		gpu_active = _batch.gpu
		gpu_name = _batch.gpu_name
		gpu_error = _batch.gpu_error
	_batch.set_genomes(flat)
	_batch.reset_state()
	# one obstacle layout + start seed per flight index (shared by all candidates = fair comparison)
	var ep_obs: Array = []
	for e in episodes:
		var o := obstacles.copy()
		var r := RandomNumberGenerator.new()
		r.seed = 555 + 31 * gen + e
		if random_obstacles > 0:
			var keep: Array = [Vector3(0, 0, 0), scene_target] if fixed_target else [Vector3(0, 0, 0)]
			o.random_fill(r, int(round(difficulty * random_obstacles)), keep, spec.ground_only())
		ep_obs.append(o)
	_agents.clear()
	for a in A:
		var e := a % episodes
		var ag := Agent.new()
		ag.rng.seed = 1000003 * gen + 7919 * e + 17
		ag.body = spec.make(ag.rng.randi())
		ag.obs = ep_obs[e]
		ag.body.reset(ag.rng, difficulty, ag.obs)
		ag.sens.resize(_S)
		_agents.append(ag)
	_period = int(ag_period())
	var ticks := int(duration / Body.CTRL_DT)
	_total_ticks = ticks
	var chunks := clampi(OS.get_processor_count() * 2, 1, A)
	_chunk = (A + chunks - 1) / chunks
	chunks = (A + _chunk - 1) / _chunk
	for t in ticks:
		_tick = t
		var gid := WorkerThreadPool.add_group_task(_sense_chunk, chunks, -1, true, "sense")
		WorkerThreadPool.wait_for_group_task_completion(gid)
		var sens := PackedFloat32Array()
		var any := false
		for ag in _agents:
			sens.append_array(ag.sens)
			any = any or ag.alive
		if not any:
			break
		_outs = _batch.step(sens)
		gid = WorkerThreadPool.add_group_task(_phys_chunk, chunks, -1, true, "physics")
		WorkerThreadPool.wait_for_group_task_completion(gid)
	var fit := PackedFloat32Array()
	fit.resize(n_gen)
	for a in A:
		fit[a / episodes] += _agents[a].score / ticks / episodes
	for i in n_gen:
		if not is_finite(fit[i]):
			fit[i] = -1.0
	return fit

func ag_period() -> float:
	return (_agents[0] as Agent).body.retarget_period() / Body.CTRL_DT

func step_generation() -> Dictionary:
	var t0 := Time.get_ticks_msec()
	var dim := theta.size()
	var half := pop / 2
	var rng := RandomNumberGenerator.new()
	rng.seed = 991 * (gen + 1) + 13
	var eps: Array = []
	var flat := PackedFloat32Array()
	flat.resize((pop + 1) * dim)
	for i in half:
		var e := PackedFloat32Array()
		e.resize(dim)
		for k in dim:
			e[k] = rng.randfn()
			flat[i * dim + k] = theta[k] + sigma * e[k]
			flat[(i + half) * dim + k] = theta[k] - sigma * e[k]
		eps.append(e)
	for k in dim:
		flat[pop * dim + k] = theta[k]        # last candidate = the un-perturbed policy
	var fit := _evaluate(flat, pop + 1)
	var order: Array = range(pop)
	order.sort_custom(func(a: int, b: int) -> bool: return fit[a] < fit[b])
	var util := PackedFloat32Array()
	util.resize(pop)
	for rank in pop:
		util[order[rank]] = float(rank) / (pop - 1) - 0.5
	var grad := PackedFloat32Array()
	grad.resize(dim)
	for i in half:
		var u := util[i] - util[i + half]
		var e: PackedFloat32Array = eps[i]
		for k in dim:
			grad[k] += u * e[k]
	_t += 1
	var c1 := 1.0 - pow(0.9, _t)
	var c2 := 1.0 - pow(0.999, _t)
	for k in dim:
		var gk := grad[k] / (pop * sigma) - 0.002 * theta[k]
		_m[k] = 0.9 * _m[k] + 0.1 * gk
		_v[k] = 0.999 * _v[k] + 0.001 * gk * gk
		theta[k] += lr * (_m[k] / c1) / (sqrt(_v[k] / c2) + 1e-8)
	var mean := 0.0
	var best := -INF
	for i in pop:
		mean += fit[i] / pop
		best = maxf(best, fit[i])
	var center := fit[pop]
	if center > spec.make(0).success_threshold() and difficulty < 1.0:
		difficulty = minf(1.0, difficulty + 0.1)
	gen += 1
	return {"gen": gen, "center": center, "mean": mean, "best": best, "difficulty": difficulty,
		"theta": theta.duplicate(), "ms": Time.get_ticks_msec() - t0, "gpu": gpu_active, "device": gpu_name}
