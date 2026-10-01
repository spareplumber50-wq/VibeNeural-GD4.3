class_name BrainBatch
extends RefCounted
## Runs the fly connectome for a whole batch of agents (population x flights) at once:
## on the GPU with a compute shader when a compute device exists, otherwise on CPU threads.
## Each agent has its own neuron state and reads its candidate's genome:
##   genome = [ W_in (n_sens_neurons x S) | W_out (n_out_neurons x C) | bias (C) | log_gain ]

const INPUT_SCALE := 3.0
const OUTPUT_SCALE := 5.0

const GLSL := """
#version 450
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) restrict readonly buffer RowPtr { int row_ptr[]; };
layout(set = 0, binding = 1, std430) restrict readonly buffer EdgePre { int e_pre[]; };
layout(set = 0, binding = 2, std430) restrict readonly buffer EdgeW { float e_w[]; };
layout(set = 0, binding = 3, std430) restrict readonly buffer Alpha { float alpha[]; };
layout(set = 0, binding = 4, std430) restrict readonly buffer SensOf { int sens_of[]; };
layout(set = 0, binding = 5, std430) restrict readonly buffer OutIdx { int out_idx[]; };
layout(set = 0, binding = 6, std430) restrict readonly buffer Genome { float genome[]; };
layout(set = 0, binding = 7, std430) restrict readonly buffer Sensors { float sensors[]; };
layout(set = 0, binding = 8, std430) restrict buffer State { float xs[]; };
layout(set = 0, binding = 9, std430) restrict buffer Rates { float rs[]; };
layout(set = 0, binding = 10, std430) restrict writeonly buffer Outs { float outs[]; };
layout(push_constant, std430) uniform Params {
	int mode; int n; int n_in; int n_out;
	int S; int C; int G; int epi;
	int agents; int parity; float in_scale; float out_scale;
} p;
void main() {
	uint gid = gl_GlobalInvocationID.x;
	if (p.mode == 0) {
		if (gid >= uint(p.agents * p.n)) return;
		int a = int(gid) / p.n;
		int j = int(gid) - a * p.n;
		int gb = (a / p.epi) * p.G;
		float gain = clamp(exp(genome[gb + p.G - 1]), 0.1, 2.0);
		int r_in = (p.parity * p.agents + a) * p.n;
		int r_out = ((1 - p.parity) * p.agents + a) * p.n;
		float acc = 0.0;
		int k = sens_of[j];
		if (k >= 0) {
			float s = 0.0;
			int wb = gb + k * p.S;
			int sb = a * p.S;
			for (int i = 0; i < p.S; i++) s += genome[wb + i] * sensors[sb + i];
			acc = s * p.in_scale;
		}
		float rec = 0.0;
		int e1 = row_ptr[j + 1];
		for (int e = row_ptr[j]; e < e1; e++) rec += e_w[e] * rs[r_in + e_pre[e]];
		acc += rec * gain;
		float xv = xs[gid];
		xv += (acc - xv) * alpha[j];
		xs[gid] = xv;
		rs[r_out + j] = tanh(xv);
	} else {
		if (gid >= uint(p.agents * p.C)) return;
		int a = int(gid) / p.C;
		int m = int(gid) - a * p.C;
		int gb = (a / p.epi) * p.G;
		int r_new = ((1 - p.parity) * p.agents + a) * p.n;
		int wo = gb + p.n_in * p.S;
		float s = 0.0;
		for (int k = 0; k < p.n_out; k++) s += genome[wo + k * p.C + m] * rs[r_new + out_idx[k]];
		outs[gid] = tanh(p.out_scale * s + genome[wo + p.n_out * p.C + m]);
	}
}
"""

class Chunk:
	var a0 := 0
	var a1 := 0
	var x := PackedFloat32Array()
	var r := PackedFloat32Array()
	var r2 := PackedFloat32Array()
	var outs := PackedFloat32Array()

var brain: FlyBrain
var S := 0
var C := 0
var G := 0
var n := 0
var n_in := 0
var n_out := 0
var agents := 1
var epi := 1
var genomes := PackedFloat32Array()
var row_ptr := PackedInt32Array()
var e_pre := PackedInt32Array()
var e_w := PackedFloat32Array()
var alpha := PackedFloat32Array()
var sens_of := PackedInt32Array()
var out_idx := PackedInt32Array()

var gpu := false
var gpu_name := ""
var gpu_error := ""
var _chunks: Array[Chunk] = []
var _chunk_size := 1
var _sens_ref := PackedFloat32Array()
var _mutex := Mutex.new()
# gpu objects
var _rd: RenderingDevice
var _shader: RID
var _pipeline: RID
var _uset: RID
var _bufs := {}
var _parity := 0

static func genome_size(b: FlyBrain, s: int, c: int) -> int:
	return s * b.sensory.size() + b.output.size() * c + c + 1

static func init_genome(b: FlyBrain, s: int, c: int, rng: RandomNumberGenerator) -> PackedFloat32Array:
	var g := PackedFloat32Array()
	g.resize(genome_size(b, s, c))
	var n_i := b.sensory.size() * s
	for i in n_i:
		g[i] = rng.randfn(0.0, 0.5)
	for i in b.output.size() * c:
		g[n_i + i] = rng.randfn(0.0, 0.01)
	g[g.size() - 1] = 0.0          # log gain -> 1.0
	return g

func _init(b: FlyBrain, s: int, c: int, n_agents: int, episodes: int, dt: float, want_gpu: bool) -> void:
	brain = b
	S = s
	C = c
	n = b.n
	n_in = b.sensory.size()
	n_out = b.output.size()
	G = genome_size(b, s, c)
	agents = n_agents
	epi = maxi(1, episodes)
	# CSR by post-synaptic neuron
	row_ptr.resize(n + 1)
	for e in b.post.size():
		row_ptr[b.post[e] + 1] += 1
	for j in n:
		row_ptr[j + 1] += row_ptr[j]
	var fill := row_ptr.duplicate()
	e_pre.resize(b.pre.size())
	e_w.resize(b.pre.size())
	for e in b.pre.size():
		var slot := fill[b.post[e]]
		fill[b.post[e]] = slot + 1
		e_pre[slot] = b.pre[e]
		e_w[slot] = b.w[e]
	alpha.resize(n)
	for i in n:
		alpha[i] = 1.0 - exp(-dt / b.tau[i])
	sens_of.resize(n)
	sens_of.fill(-1)
	for k in n_in:
		sens_of[b.sensory[k]] = k
	out_idx = b.output.duplicate()
	genomes.resize(((agents + epi - 1) / epi) * G)
	if want_gpu:
		_init_gpu()
	if not gpu:
		_init_cpu()

func _init_cpu() -> void:
	var cores := clampi(OS.get_processor_count() * 2, 1, agents)
	_chunk_size = (agents + cores - 1) / cores
	_chunks.clear()
	var a := 0
	while a < agents:
		var ch := Chunk.new()
		ch.a0 = a
		ch.a1 = mini(a + _chunk_size, agents)
		ch.x.resize((ch.a1 - ch.a0) * n)
		ch.r.resize((ch.a1 - ch.a0) * n)
		ch.r2.resize((ch.a1 - ch.a0) * n)
		_chunks.append(ch)
		a += _chunk_size

func _bytes_i(a: PackedInt32Array) -> PackedByteArray:
	return (a if a.size() > 0 else PackedInt32Array([0])).to_byte_array()

func _bytes_f(a: PackedFloat32Array) -> PackedByteArray:
	return (a if a.size() > 0 else PackedFloat32Array([0.0])).to_byte_array()

func _init_gpu() -> void:
	_rd = RenderingServer.create_local_rendering_device()
	if _rd == null:
		gpu_error = "no compute device (needs the Forward+/Mobile renderer and a Vulkan GPU)"
		return
	var src := RDShaderSource.new()
	src.language = RenderingDevice.SHADER_LANGUAGE_GLSL
	src.source_compute = GLSL
	var spirv := _rd.shader_compile_spirv_from_source(src)
	if spirv.compile_error_compute != "":
		gpu_error = "shader: " + spirv.compile_error_compute
		_release_rd()
		return
	_shader = _rd.shader_create_from_spirv(spirv)
	if not _shader.is_valid():
		gpu_error = "could not create shader"
		_release_rd()
		return
	_pipeline = _rd.compute_pipeline_create(_shader)
	var zero_state := PackedFloat32Array()
	zero_state.resize(agents * n)
	var zero_r := PackedFloat32Array()
	zero_r.resize(2 * agents * n)
	var sens_zero := PackedFloat32Array()
	sens_zero.resize(agents * S)
	var out_zero := PackedFloat32Array()
	out_zero.resize(agents * C)
	var data := [_bytes_i(row_ptr), _bytes_i(e_pre), _bytes_f(e_w), _bytes_f(alpha), _bytes_i(sens_of), _bytes_i(out_idx),
		_bytes_f(genomes), _bytes_f(sens_zero), _bytes_f(zero_state), _bytes_f(zero_r), _bytes_f(out_zero)]
	var uniforms: Array[RDUniform] = []
	for i in data.size():
		var buf := _rd.storage_buffer_create(data[i].size(), data[i])
		_bufs[i] = buf
		var u := RDUniform.new()
		u.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		u.binding = i
		u.add_id(buf)
		uniforms.append(u)
	_uset = _rd.uniform_set_create(uniforms, _shader, 0)
	gpu = _uset.is_valid()
	if gpu:
		gpu_name = _rd.get_device_name()
	else:
		gpu_error = "could not bind buffers"
		release()

func _release_rd() -> void:
	if _rd != null:
		_rd.free()
		_rd = null

## free GPU resources (call on the same thread that created the batch)
func release() -> void:
	if _rd != null:
		for k in _bufs:
			if _bufs[k].is_valid():
				_rd.free_rid(_bufs[k])
		_bufs.clear()
		if _pipeline.is_valid():
			_rd.free_rid(_pipeline)
		if _shader.is_valid():
			_rd.free_rid(_shader)
		_release_rd()
	gpu = false

func set_genomes(g: PackedFloat32Array) -> void:
	genomes = g
	if gpu:
		var b := _bytes_f(g)
		_rd.buffer_update(_bufs[6], 0, b.size(), b)

func reset_state() -> void:
	_parity = 0
	if gpu:
		var z := PackedFloat32Array()
		z.resize(agents * n)
		var zb := z.to_byte_array()
		_rd.buffer_update(_bufs[8], 0, zb.size(), zb)
		var z2 := PackedFloat32Array()
		z2.resize(2 * agents * n)
		var zb2 := z2.to_byte_array()
		_rd.buffer_update(_bufs[9], 0, zb2.size(), zb2)
	else:
		for ch in _chunks:
			ch.x.fill(0.0)
			ch.r.fill(0.0)
			ch.r2.fill(0.0)

## sens: agents * S floats -> returns agents * C motor / stick outputs in (-1, 1)
func step(sens: PackedFloat32Array) -> PackedFloat32Array:
	if gpu:
		return _step_gpu(sens)
	_sens_ref = sens
	if _chunks.size() > 1:
		var gid := WorkerThreadPool.add_group_task(_cpu_task, _chunks.size(), -1, true, "brain")
		WorkerThreadPool.wait_for_group_task_completion(gid)
	else:
		_cpu_task(0)
	var out := PackedFloat32Array()
	for ch in _chunks:
		out.append_array(ch.outs)
	return out

func _step_gpu(sens: PackedFloat32Array) -> PackedFloat32Array:
	var sb := sens.to_byte_array()
	_rd.buffer_update(_bufs[7], 0, sb.size(), sb)
	var cl := _rd.compute_list_begin()
	_rd.compute_list_bind_compute_pipeline(cl, _pipeline)
	_rd.compute_list_bind_uniform_set(cl, _uset, 0)
	var pc := PackedInt32Array([0, n, n_in, n_out, S, C, G, epi, agents, _parity, 0, 0])
	var pcb := pc.to_byte_array()
	pcb.encode_float(40, INPUT_SCALE)
	pcb.encode_float(44, OUTPUT_SCALE)
	_rd.compute_list_set_push_constant(cl, pcb, pcb.size())
	_rd.compute_list_dispatch(cl, (agents * n + 63) / 64, 1, 1)
	_rd.compute_list_add_barrier(cl)
	pcb.encode_s32(0, 1)
	_rd.compute_list_set_push_constant(cl, pcb, pcb.size())
	_rd.compute_list_dispatch(cl, (agents * C + 63) / 64, 1, 1)
	_rd.compute_list_end()
	_rd.submit()
	_rd.sync()
	_parity = 1 - _parity
	return _rd.buffer_get_data(_bufs[10]).to_float32_array()

func _cpu_task(ci: int) -> void:
	var ch := _chunks[ci]
	var sens := _sens_ref
	var na := ch.a1 - ch.a0
	ch.outs.resize(na * C)
	for ai in na:
		var a := ch.a0 + ai
		var gb := (a / epi) * G
		var gain := clampf(exp(genomes[gb + G - 1]), 0.1, 2.0)
		var base := ai * n
		for j in n:
			var acc := 0.0
			var k := sens_of[j]
			if k >= 0:
				var s := 0.0
				var wb := gb + k * S
				var sb := a * S
				for i in S:
					s += genomes[wb + i] * sens[sb + i]
				acc = s * INPUT_SCALE
			var rec := 0.0
			for e in range(row_ptr[j], row_ptr[j + 1]):
				rec += e_w[e] * ch.r[base + e_pre[e]]
			acc += rec * gain
			var xv := ch.x[base + j]
			xv += (acc - xv) * alpha[j]
			ch.x[base + j] = xv
			ch.r2[base + j] = tanh(xv)
	var t := ch.r
	ch.r = ch.r2
	ch.r2 = t
	for ai in na:
		var a := ch.a0 + ai
		var gb := (a / epi) * G
		var wo := gb + n_in * S
		for m in C:
			var s := 0.0
			for k in n_out:
				s += genomes[wo + k * C + m] * ch.r[ai * n + out_idx[k]]
			ch.outs[ai * C + m] = tanh(OUTPUT_SCALE * s + genomes[wo + n_out * C + m])

## neuron rates of one agent (for the brain view)
func rates(agent: int) -> PackedFloat32Array:
	if gpu:
		var region := _parity * agents * n + agent * n
		return _rd.buffer_get_data(_bufs[9], region * 4, n * 4).to_float32_array()
	for ch in _chunks:
		if agent >= ch.a0 and agent < ch.a1:
			return ch.r.slice((agent - ch.a0) * n, (agent - ch.a0 + 1) * n)
	return PackedFloat32Array()
