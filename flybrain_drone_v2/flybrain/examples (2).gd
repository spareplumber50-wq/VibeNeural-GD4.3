class_name Examples
extends RefCounted
## Ready-made scenes. obstacles: ["box", cx, cy, cz, hx, hy, hz] or ["sphere", cx, cy, cz, r]

static func list() -> Array:
	return [
		{"name": "Drone: hover and chase", "desc": "Empty sky. Learn to hover, then chase a moving target.",
			"kind": "drone", "drone": {}, "obstacles": [], "range": 2.0, "random": 0, "moving": true, "fixed": false},
		{"name": "Drone: pillar slalom", "desc": "Three tall pillars between you and the target. Whiskers let the brain feel them.",
			"kind": "drone", "drone": {}, "range": 2.0, "random": 0, "moving": false, "fixed": true, "target": [0.0, 1.5, -4.0],
			"obstacles": [["box", 0.15, 2.2, -1.3, 0.15, 2.2, 0.15], ["box", -0.3, 2.2, -2.3, 0.15, 2.2, 0.15], ["box", 0.2, 2.2, -3.1, 0.15, 2.2, 0.15]]},
		{"name": "Drone: narrow gate in wind", "desc": "Fly through a 1 m gap while wind and gusts push you around.",
			"kind": "drone", "drone": {"wind": 2.0, "gust": 1.2}, "range": 2.0, "random": 0, "moving": false, "fixed": true, "target": [0.0, 1.5, -3.6],
			"obstacles": [["box", -1.15, 2.2, -2.0, 0.65, 2.2, 0.12], ["box", 1.15, 2.2, -2.0, 0.65, 2.2, 0.12]]},
		{"name": "Drone: random forest", "desc": "Every training flight gets a new random set of pillars and balls.",
			"kind": "drone", "drone": {}, "obstacles": [], "range": 3.0, "random": 6, "moving": true, "fixed": false},
		{"name": "Hexacopter in a crosswind", "desc": "Heavier six-rotor drone, steady wind, random obstacles.",
			"kind": "drone", "drone": {"motor_count": 6, "body_mass": 0.9, "max_thrust": 5.0, "arm_length": 0.22, "wind": 3.0, "gust": 0.8},
			"obstacles": [], "range": 2.5, "random": 3, "moving": true, "fixed": false},
		{"name": "Quadruped: walk to target", "desc": "Four-legged walker. Fitness rises once it learns a gait.",
			"kind": "quadruped", "walker": {}, "obstacles": [], "range": 3.0, "random": 0, "moving": true, "fixed": false},
		{"name": "Quadruped: obstacle field", "desc": "Walk to the far target between pillars.",
			"kind": "quadruped", "walker": {}, "range": 3.0, "random": 0, "moving": false, "fixed": true, "target": [0.0, 0.0, -4.5],
			"obstacles": [["box", 0.55, 0.5, -1.6, 0.15, 0.5, 0.15], ["box", -0.55, 0.5, -2.6, 0.15, 0.5, 0.15], ["box", 0.3, 0.5, -3.6, 0.15, 0.5, 0.15]]},
		{"name": "Biped: walk to target", "desc": "Two legs. Starts with training wheels that fade as it improves.",
			"kind": "biped", "walker": {}, "obstacles": [], "range": 2.5, "random": 0, "moving": true, "fixed": false},
	]
