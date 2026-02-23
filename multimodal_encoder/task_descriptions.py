"""DMC task description registry.

Maps (domain, task) pairs to textual descriptions for language-conditioned RL.
"""

TASK_DESCRIPTIONS = {
    # Cartpole
    ("cartpole", "swingup"): "Swing up the pole from the bottom position to upright and balance it vertically on the cart.",
    ("cartpole", "balance"): "Keep the pole balanced upright on the cart by moving the cart left and right.",
    ("cartpole", "balance_sparse"): "Keep the pole balanced upright on the cart to earn reward.",
    ("cartpole", "swingup_sparse"): "Swing the pole up to vertical and balance it on the cart to earn reward.",
    # Walker
    ("walker", "walk"): "Make the bipedal walker walk forward as fast as possible while staying upright. NOTE: Focus on the main body and not on floor or background",
    ("walker", "run"): "Make the bipedal walker run forward at high speed while maintaining balance.",
    ("walker", "stand"): "Keep the bipedal walker standing upright without falling.",
    # Cheetah
    ("cheetah", "run"): "Make the half-cheetah run forward as fast as possible.",
    # Reacher
    ("reacher", "easy"): "Move the two-link reacher arm so that the fingertip reaches the target location.",
    ("reacher", "hard"): "Move the two-link reacher arm to precisely reach a distant target location.",
    # Finger
    ("finger", "spin"): "Rotate the object on the finger by applying torque to the finger joints.",
    ("finger", "turn_easy"): "Rotate the object on the finger to a target orientation.",
    ("finger", "turn_hard"): "Precisely rotate the object on the finger to a target orientation.",
    # Hopper
    ("hopper", "hop"): "Make the one-legged hopper hop forward by jumping and landing repeatedly.",
    ("hopper", "stand"): "Keep the one-legged hopper balanced in a standing position.",
    # Humanoid
    ("humanoid", "walk"): "Control the humanoid to walk forward on two legs while maintaining upright posture.",
    ("humanoid", "run"): "Control the humanoid to run forward on two legs at high speed.",
    ("humanoid", "stand"): "Keep the humanoid balanced in a standing position.",
    # Cup
    ("cup", "catch"): "Swing the ball attached by a string to the cup and catch it inside the cup.",
    # Pendulum
    ("pendulum", "swingup"): "Swing up the pendulum from the bottom to the upright position and balance it.",
    # Acrobot
    ("acrobot", "swingup"): "Swing up the two-link acrobot to the upright position by actuating the middle joint.",
    ("acrobot", "swingup_sparse"): "Swing the two-link acrobot upright to earn reward.",
    # Quadruped
    ("quadruped", "walk"): "Make the four-legged robot walk forward while maintaining balance.",
    ("quadruped", "run"): "Make the four-legged robot run forward at high speed.",
    # Fish
    ("fish", "swim"): "Control the fish to swim toward the target location in the fluid.",
    ("fish", "upright"): "Keep the fish in an upright orientation while swimming.",
    # Swimmer
    ("swimmer", "swimmer6"): "Control the six-link swimmer to move forward through the fluid.",
    ("swimmer", "swimmer15"): "Control the fifteen-link swimmer to move forward through the fluid.",
}


def get_task_description(domain: str, task: str) -> str:
    """Look up the task description for a DMC (domain, task) pair.

    Falls back to a generic description if no specific entry exists.
    """
    key = (domain, task)
    if key in TASK_DESCRIPTIONS:
        return TASK_DESCRIPTIONS[key]
    return f"In the {domain} environment, accomplish the {task} task successfully."


def get_task_description_from_name(task_name: str) -> str:
    """Extract domain/task from an env name like 'walker_walk' and return description.

    Handles special cases like 'finger_turn_easy' where the task contains underscores.
    """
    if "sparse" in task_name or "finger_turn" in task_name:
        _name, difficulty = task_name.rsplit("_", 1)
        domain, task = _name.rsplit("_", 1)
        task = task + "_" + difficulty
    else:
        domain, task = task_name.rsplit("_", 1)
    return get_task_description(domain, task)
