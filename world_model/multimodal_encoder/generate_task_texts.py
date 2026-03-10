import json, random, os

random.seed(42)

# ============================================================
# 10 TASK-IRRELEVANT CATEGORIES
# Each has 15 templates explaining WHY that category is irrelevant.
# Templates use {agent} and {task_aspect} placeholders for task-specific fill.
# ============================================================

IRRELEVANT = {
    "background_color": [
        "The background color has no effect on {agent}'s dynamics or control.",
        "Changes to the background gradient do not alter {agent}'s {task_aspect}.",
        "The sky color behind the scene provides no useful signal for {task_aspect}.",
        "Background hue variations are irrelevant to how {agent} should be controlled.",
        "The color of the distant backdrop does not influence {agent}'s joint torques.",
        "Whether the background is blue or black has no bearing on {task_aspect}.",
        "The background gradient carries no information about {agent}'s body state.",
        "Pixel colors in the background region are uninformative for {task_aspect}.",
        "The backdrop appearance does not correlate with {agent}'s reward signal.",
        "No control-relevant information exists in the background color of the scene.",
        "The background color cannot help predict {agent}'s next state or reward.",
        "Attending to the background color would waste representation capacity for {task_aspect}.",
        "The background provides only static visual noise unrelated to {agent}'s motion.",
        "The background hue is decorative and irrelevant to {agent}'s physics.",
        "Scene background colors do not encode any dynamics useful for controlling {agent}.",
    ],
    "floor_texture": [
        "The floor texture pattern does not affect {agent}'s contact dynamics.",
        "Whether the ground has tiles or wood grain is irrelevant to {task_aspect}.",
        "The visual pattern on the floor carries no physics information for {agent}.",
        "Floor texture is purely cosmetic and does not change {agent}'s friction or control.",
        "The ground surface pattern provides no signal about {agent}'s joint states.",
        "Attending to floor texture details wastes capacity needed for {task_aspect}.",
        "The tile or grid pattern on the floor does not influence {agent}'s reward.",
        "Floor visual patterns are static decorations unrelated to {agent}'s dynamics.",
        "The texture of the ground surface cannot help predict {agent}'s balance or motion.",
        "No control policy should depend on the floor's visual texture pattern.",
        "The floor pattern is irrelevant because {agent}'s physics are independent of it.",
        "Ground texture variations do not encode useful information for {task_aspect}.",
        "The visual appearance of the floor has no bearing on {agent}'s actions.",
        "Floor pattern details are distractors that do not help with {task_aspect}.",
        "The ground's visual texture is unrelated to the forces acting on {agent}.",
    ],
    "floor_color": [
        "The color of the floor does not affect {agent}'s dynamics or reward.",
        "Whether the ground is gray or green has no bearing on {task_aspect}.",
        "Floor color carries no information about {agent}'s joint angles or velocities.",
        "The ground color is purely aesthetic and irrelevant to controlling {agent}.",
        "Attending to floor color would not improve {agent}'s performance at {task_aspect}.",
        "The floor color does not correlate with any aspect of {agent}'s physics.",
        "A different floor color would not change how {agent} should be controlled.",
        "The ground surface color is uninformative for predicting {agent}'s state transitions.",
        "Floor color is a visual distractor with no relevance to {task_aspect}.",
        "No reward signal depends on the color of the ground beneath {agent}.",
        "The floor's color provides zero bits of information about {agent}'s control needs.",
        "Ground color variations are cosmetic and should be ignored for {task_aspect}.",
        "The floor color cannot help distinguish good from bad actions for {agent}.",
        "Whether the floor is dark or light does not affect {agent}'s task performance.",
        "The color of the ground is independent of {agent}'s dynamics and objectives.",
    ],
    "agent_color": [
        "The color of {agent}'s body does not affect its dynamics or control.",
        "Whether {agent} is orange or blue has no bearing on {task_aspect}.",
        "The agent's surface color carries no information about joint torques or forces.",
        "Body color is purely visual and irrelevant to how {agent} should act.",
        "Attending to {agent}'s color would not help predict its next state.",
        "The rendering color of {agent} does not influence its reward for {task_aspect}.",
        "A red or green {agent} would behave identically under the same control.",
        "{agent}'s body color is decorative and provides no control-relevant signal.",
        "The hue of {agent}'s body segments is uninformative for {task_aspect}.",
        "No physics or reward information is encoded in {agent}'s surface color.",
        "The agent's color does not correlate with any aspect of {task_aspect}.",
        "{agent}'s color is irrelevant because the same dynamics apply regardless of color.",
        "Changing {agent}'s color would not change optimal actions for {task_aspect}.",
        "The visual color of {agent} should be ignored when learning to control it.",
        "Body color variations contain no useful signal about {agent}'s motion or balance.",
    ],
    "agent_material": [
        "The visual material finish of {agent} does not reflect its actual physics.",
        "Whether {agent} looks metallic or matte has no effect on {task_aspect}.",
        "The rendered surface material is cosmetic and unrelated to {agent}'s dynamics.",
        "A glossy or matte finish on {agent} does not change its joint behavior.",
        "The visual texture of {agent}'s surface carries no control-relevant information.",
        "Attending to {agent}'s material appearance would waste capacity for {task_aspect}.",
        "The surface finish does not influence the forces or torques acting on {agent}.",
        "Whether {agent} appears chrome or plastic has no bearing on its control.",
        "Material appearance is a rendering choice irrelevant to {agent}'s {task_aspect}.",
        "The shininess of {agent}'s body provides no information about its state.",
        "No reward or dynamics depend on the visual material of {agent}'s surface.",
        "{agent}'s material look is purely aesthetic and not useful for {task_aspect}.",
        "The surface finish of {agent} does not encode any physics information.",
        "A rubber or metal appearance on {agent} would not change its optimal policy.",
        "Visual material properties of {agent} are decorative and should be ignored.",
    ],
    "lighting": [
        "The lighting direction does not affect {agent}'s dynamics or control policy.",
        "Whether light comes from left or right has no bearing on {task_aspect}.",
        "Lighting color and intensity carry no information about {agent}'s joint states.",
        "The illumination setup is purely visual and irrelevant to controlling {agent}.",
        "Attending to lighting conditions would not improve {agent}'s {task_aspect}.",
        "Light direction does not correlate with {agent}'s reward or state transitions.",
        "The scene's lighting setup provides no useful signal about {agent}'s physics.",
        "Whether illumination is warm or cool has no effect on {agent}'s control.",
        "Lighting variations are rendering choices irrelevant to {agent}'s actions.",
        "The brightness or color of lights does not influence how {agent} should act.",
        "No control-relevant information exists in the scene's lighting configuration.",
        "The lighting angle cannot help predict {agent}'s balance or motion state.",
        "Scene illumination is decorative and unrelated to {task_aspect}.",
        "Different lighting setups would not change the optimal policy for {agent}.",
        "Light intensity and direction are visual distractors for {task_aspect}.",
    ],
    "joint_markers": [
        "Visual joint markers are rendering overlays with no physics meaning for {agent}.",
        "Whether joints are highlighted or not does not affect {agent}'s {task_aspect}.",
        "Joint marker colors and shapes carry no information about actual torques.",
        "The visual indicators at joints are decorative and irrelevant to {agent}'s control.",
        "Attending to joint marker appearance would not help predict {agent}'s dynamics.",
        "Joint highlight colors do not encode the magnitude of forces on {agent}.",
        "The presence or absence of joint markers has no bearing on {task_aspect}.",
        "Visual joint annotations are cosmetic overlays unrelated to {agent}'s state.",
        "Joint marker shapes cannot help determine optimal actions for {agent}.",
        "The color of joint indicators provides no information about {agent}'s motion.",
        "Whether joints show yellow spheres or no markers is irrelevant to control.",
        "Joint visual markers do not correlate with {agent}'s reward signal.",
        "The rendering of joint locations is purely aesthetic for {task_aspect}.",
        "No control policy should depend on the visual style of joint markers.",
        "Joint marker appearance is a display choice unrelated to {agent}'s physics.",
    ],
    "shadow": [
        "Shadows on the floor are visual effects that do not affect {agent}'s dynamics.",
        "The shadow direction does not carry information about {agent}'s {task_aspect}.",
        "Whether shadows are sharp or soft has no bearing on {agent}'s control.",
        "Shadow appearance is a rendering artifact irrelevant to {agent}'s physics.",
        "Attending to shadow patterns would waste representation capacity for {task_aspect}.",
        "The shadow beneath {agent} does not encode its joint states or velocities.",
        "Shadow color and shape cannot help predict {agent}'s next state or reward.",
        "The presence or absence of shadows is irrelevant to how {agent} should act.",
        "Shadow length and direction are lighting artifacts unrelated to {agent}'s motion.",
        "No reward signal depends on the visual appearance of {agent}'s shadow.",
        "The shadow is a visual byproduct providing no useful signal for {task_aspect}.",
        "Whether the shadow is long or short does not affect {agent}'s optimal actions.",
        "Shadow darkness does not correlate with {agent}'s balance or performance.",
        "The ground shadow is purely cosmetic and unrelated to {agent}'s control needs.",
        "Shadows provide no information about the forces or torques acting on {agent}.",
    ],
    "camera": [
        "The camera viewpoint is a rendering choice that does not affect {agent}'s physics.",
        "Camera angle has no bearing on {agent}'s dynamics or optimal {task_aspect}.",
        "Whether the camera is close or far does not change {agent}'s control policy.",
        "The camera position does not encode information about {agent}'s joint states.",
        "Attending to camera viewpoint would not improve {agent}'s {task_aspect} performance.",
        "A side view or front view does not alter the forces acting on {agent}.",
        "Camera zoom level is irrelevant to {agent}'s reward or state transitions.",
        "The camera angle cannot help determine what actions {agent} should take.",
        "Whether the camera is above or beside {agent} has no effect on control.",
        "Camera position is an observation property unrelated to {agent}'s {task_aspect}.",
        "No control-relevant physics information depends on the camera's viewing angle.",
        "The camera distance from {agent} does not influence its dynamics or reward.",
        "A different camera perspective would not change how {agent} should be controlled.",
        "Camera orientation is a display choice irrelevant to {agent}'s action selection.",
        "The viewing angle provides no useful information about {agent}'s body dynamics.",
    ],
    "atmosphere": [
        "Atmospheric effects like fog do not influence {agent}'s dynamics or control.",
        "Haze or particles in the air have no bearing on {agent}'s {task_aspect}.",
        "Visual atmosphere effects are rendering choices irrelevant to {agent}'s physics.",
        "Whether the scene has fog or is clear does not affect {agent}'s reward.",
        "Attending to atmospheric haze would waste capacity needed for {task_aspect}.",
        "Dust particles in the scene carry no information about {agent}'s joint states.",
        "Atmospheric effects do not correlate with {agent}'s balance or motion state.",
        "The presence of fog or mist does not change how {agent} should be controlled.",
        "Visual bloom or glow effects are decorative and irrelevant to {task_aspect}.",
        "No control policy should attend to atmospheric particle effects in the scene.",
        "Atmospheric conditions in the rendering do not affect {agent}'s dynamics.",
        "Scene haze provides no useful signal for predicting {agent}'s state transitions.",
        "Whether the air is clear or foggy has no bearing on {agent}'s performance.",
        "Atmospheric rendering effects cannot help optimize {agent}'s {task_aspect}.",
        "Visual atmosphere is purely cosmetic and unrelated to {agent}'s control problem.",
    ],
}

IRRELEVANT_CATEGORIES = list(IRRELEVANT.keys())

# ============================================================
# DMC TASKS
# ============================================================

TASKS = {
    ("walker", "stand"): {
        "agent": "the bipedal walker",
        "task_aspect": "upright balance",
        "goals": [
            "Make the bipedal walker stand upright without falling.",
            "Keep the bipedal walker balanced in a standing position.",
            "The bipedal walker must remain standing and upright.",
            "Maintain an upright standing posture with the bipedal walker.",
            "Stabilize the bipedal walker in a standing pose.",
            "Control the bipedal walker to hold a steady upright stance.",
            "The walker should stay balanced on its feet without toppling.",
            "Prevent the bipedal walker from falling while standing still.",
            "The goal is to keep the bipedal walker erect and balanced.",
            "The bipedal walker must stand on its legs without collapsing.",
        ],
    },
    ("walker", "walk"): {
        "agent": "the bipedal walker",
        "task_aspect": "forward walking",
        "goals": [
            "Make the bipedal walker walk forward as fast as possible.",
            "The bipedal walker must walk forward while staying upright.",
            "Control the walker to locomote forward with a walking gait.",
            "Drive the bipedal walker forward by coordinating its legs.",
            "The walker should stride forward smoothly on two legs.",
            "Achieve fast forward walking with the bipedal walker.",
            "The bipedal walker must advance forward by walking.",
            "Move the bipedal walker forward using alternating leg steps.",
            "Walk the bipedal walker forward as quickly as possible.",
            "The goal is fast upright forward walking with the walker.",
        ],
    },
    ("walker", "run"): {
        "agent": "the bipedal walker",
        "task_aspect": "high-speed running",
        "goals": [
            "Make the bipedal walker run forward at maximum speed.",
            "The bipedal walker must sprint forward as fast as possible.",
            "Control the walker to achieve high-speed forward running.",
            "Drive the bipedal walker to run forward without falling.",
            "The walker should run at top speed while staying upright.",
            "Achieve the fastest possible forward running with the walker.",
            "The bipedal walker must move forward at sprinting speed.",
            "Make the walker run forward with explosive fast steps.",
            "Maximize the bipedal walker's forward running velocity.",
            "The goal is maximum speed forward running for the walker.",
        ],
    },
    ("cheetah", "run"): {
        "agent": "the half-cheetah",
        "task_aspect": "forward running speed",
        "goals": [
            "Make the half-cheetah run forward as fast as possible.",
            "The half-cheetah must sprint forward at maximum velocity.",
            "Control the cheetah to achieve top forward running speed.",
            "Drive the half-cheetah forward with coordinated leg motion.",
            "The cheetah should gallop forward as rapidly as possible.",
            "Achieve maximum forward velocity with the half-cheetah.",
            "The half-cheetah must move forward at its fastest speed.",
            "Make the cheetah run forward using all its leg joints.",
            "Maximize the half-cheetah's forward running speed.",
            "The goal is the fastest forward running for the cheetah.",
        ],
    },
    ("cartpole", "balance"): {
        "agent": "the cart-pole",
        "task_aspect": "pole balancing",
        "goals": [
            "Balance the pole upright on the cart.",
            "Keep the pole vertical by moving the cart left and right.",
            "The cart must move to prevent the pole from falling.",
            "Maintain the pole in an upright balanced position on the cart.",
            "Stabilize the inverted pole on the moving cart.",
            "Control the cart to keep the pole from toppling over.",
            "The pole must stay balanced vertically above the cart.",
            "Use cart motion to maintain the pole's upright balance.",
            "Prevent the pole from falling by adjusting the cart position.",
            "The goal is to keep the pole balanced upright on the cart.",
        ],
    },
    ("cartpole", "swingup"): {
        "agent": "the cart-pole",
        "task_aspect": "swing-up and balance",
        "goals": [
            "Swing the pole up from below and balance it upright.",
            "Bring the hanging pole to vertical and stabilize it.",
            "The cart must swing the pole from down to up and hold it.",
            "Pump energy into the pole to swing it upright then balance.",
            "Invert the pole from its hanging position and stabilize.",
            "Control the cart to swing the pole up to vertical balance.",
            "The pole starts hanging down and must be swung to upright.",
            "Use cart oscillations to swing the pole up and balance it.",
            "Swing up the pendulum on the cart then hold it vertical.",
            "The goal is to swing the pole up and maintain balance.",
        ],
    },
    ("cartpole", "balance_sparse"): {
        "agent": "the cart-pole",
        "task_aspect": "precise pole balancing",
        "goals": [
            "Balance the pole precisely upright for sparse reward.",
            "Keep the pole exactly vertical on the cart.",
            "The pole must be held within a tight angle of vertical.",
            "Maintain strict upright balance of the pole on the cart.",
            "Precisely stabilize the pole at its vertical position.",
            "Control the cart to hold the pole exactly upright.",
            "The pole must remain within a narrow upright tolerance.",
            "Achieve precise vertical balance of the pole on the cart.",
            "The cart must keep the pole at exact vertical balance.",
            "The goal is precise pole balance for sparse reward signal.",
        ],
    },
    ("cartpole", "swingup_sparse"): {
        "agent": "the cart-pole",
        "task_aspect": "precise swing-up balance",
        "goals": [
            "Swing the pole up and hold it precisely upright.",
            "Bring the pole from hanging to exact vertical position.",
            "The cart must swing up the pole and hold it exactly upright.",
            "Pump the pole to vertical and achieve precise balance.",
            "Invert the hanging pole and stabilize it precisely.",
            "Control the cart to swing up and exactly balance the pole.",
            "The pole must be swung to precise upright for sparse reward.",
            "Swing the pole up then maintain exact vertical position.",
            "Achieve precise inverted balance after swinging the pole up.",
            "The goal is exact pole inversion with sparse reward signal.",
        ],
    },
    ("reacher", "easy"): {
        "agent": "the reacher arm",
        "task_aspect": "target reaching",
        "goals": [
            "Move the reacher fingertip to the target location.",
            "The planar arm must reach and touch the target.",
            "Position the end-effector at the target using joint control.",
            "Control the two-link arm to reach the nearby target.",
            "The reacher fingertip should arrive at the goal position.",
            "Navigate the arm's tip to the target point on the plane.",
            "Reach the target location with the arm's fingertip.",
            "The planar reacher must place its tip on the target.",
            "Move both arm joints to bring the fingertip to the target.",
            "The goal is to position the reacher tip at the target.",
        ],
    },
    ("reacher", "hard"): {
        "agent": "the reacher arm",
        "task_aspect": "precise distant reaching",
        "goals": [
            "Reach the distant target with the arm's fingertip.",
            "The planar arm must reach a challenging target location.",
            "Position the end-effector precisely at the far target.",
            "Control the two-link arm to reach a hard target position.",
            "The reacher tip must reach a target near the workspace edge.",
            "Navigate the arm to a difficult-to-reach goal position.",
            "Reach the challenging target location with the fingertip.",
            "The planar reacher must extend to a distant target point.",
            "Move the arm joints to reach a far or awkward target.",
            "The goal is reaching a hard target at the workspace limit.",
        ],
    },
    ("finger", "spin"): {
        "agent": "the finger",
        "task_aspect": "object spinning",
        "goals": [
            "Use the finger to spin the object as fast as possible.",
            "Make the spinner rotate continuously using the finger.",
            "The finger must keep the object spinning at high speed.",
            "Drive continuous fast rotation of the spinner with the finger.",
            "Spin the object rapidly by applying force with the finger.",
            "Control the finger to maintain fast object rotation.",
            "The spinner should rotate as quickly and steadily as possible.",
            "Use finger contact to sustain rapid spinning of the object.",
            "Maximize the angular velocity of the spinning object.",
            "The goal is continuous fast spinning of the object.",
        ],
    },
    ("finger", "turn_easy"): {
        "agent": "the finger",
        "task_aspect": "angle matching",
        "goals": [
            "Turn the object to match the target angle using the finger.",
            "Rotate the object to the goal orientation with the finger.",
            "The finger must orient the object to the target angle.",
            "Control the finger to set the object to the desired angle.",
            "Adjust the object rotation to match the target using the finger.",
            "Turn the rotatable object to the specified goal angle.",
            "The object must be rotated to the correct target orientation.",
            "Use the finger to precisely orient the object to the target.",
            "Rotate the body to its target angle with finger contact.",
            "The goal is turning the object to the target angle.",
        ],
    },
    ("finger", "turn_hard"): {
        "agent": "the finger",
        "task_aspect": "precise angle matching",
        "goals": [
            "Precisely turn the object to a difficult target angle.",
            "Rotate the object to a challenging goal orientation.",
            "The finger must achieve exact angular positioning of the object.",
            "Control the finger for tight-tolerance angle matching.",
            "Set the object to a demanding target angle with the finger.",
            "Turn the object precisely to a hard-to-reach orientation.",
            "The object must be rotated exactly to the precise target.",
            "Use the finger for fine angular control to match the goal.",
            "Achieve precise object orientation at a difficult target angle.",
            "The goal is exact angular positioning with tight tolerance.",
        ],
    },
    ("hopper", "stand"): {
        "agent": "the hopper",
        "task_aspect": "single-leg balance",
        "goals": [
            "Make the one-legged hopper stand upright and balanced.",
            "The hopper must maintain a standing upright position.",
            "Keep the hopper balanced on its single leg without falling.",
            "Stabilize the one-legged hopper in a standing pose.",
            "The hopper should hold an erect balanced posture.",
            "Control the hopper to stand still on one leg.",
            "Maintain upright balance with the one-legged hopper.",
            "The hopper must stay standing without toppling over.",
            "Keep the hopper's body erect and balanced on its leg.",
            "The goal is stable upright standing for the hopper.",
        ],
    },
    ("hopper", "hop"): {
        "agent": "the hopper",
        "task_aspect": "forward hopping",
        "goals": [
            "Make the one-legged hopper hop forward as fast as possible.",
            "The hopper must bounce forward rapidly on its single leg.",
            "Control the hopper to hop forward with maximum speed.",
            "Drive the hopper forward through repeated jumping motions.",
            "The hopper should advance forward by hopping quickly.",
            "Achieve fast forward hopping with the one-legged hopper.",
            "The hopper must propel itself forward through hops.",
            "Make the hopper jump forward repeatedly at high speed.",
            "Maximize forward hopping velocity of the one-legged hopper.",
            "The goal is fast forward locomotion by hopping.",
        ],
    },
    ("quadruped", "walk"): {
        "agent": "the quadruped",
        "task_aspect": "four-legged walking",
        "goals": [
            "Make the quadruped walk forward on all four legs.",
            "The four-legged agent must walk forward with a steady gait.",
            "Control the quadruped to advance forward by walking.",
            "Drive the quadruped forward using coordinated leg motion.",
            "The quadruped should walk forward with balanced strides.",
            "Achieve forward walking with the four-legged quadruped.",
            "The quadruped must locomote forward on its four legs.",
            "Walk the quadruped forward as quickly as possible.",
            "Coordinate all four legs to walk the quadruped forward.",
            "The goal is steady forward walking for the quadruped.",
        ],
    },
    ("quadruped", "run"): {
        "agent": "the quadruped",
        "task_aspect": "four-legged running",
        "goals": [
            "Make the quadruped run forward at maximum speed.",
            "The four-legged agent must gallop forward as fast as possible.",
            "Control the quadruped to sprint forward on all fours.",
            "Drive the quadruped forward at top running speed.",
            "The quadruped should run forward at its fastest velocity.",
            "Achieve maximum forward running speed with the quadruped.",
            "The quadruped must sprint forward using all four legs.",
            "Run the quadruped forward as rapidly as possible.",
            "Maximize the quadruped's forward velocity by running.",
            "The goal is fastest forward running for the quadruped.",
        ],
    },
    ("cup", "catch"): {
        "agent": "the cup-and-ball",
        "task_aspect": "ball catching",
        "goals": [
            "Swing the ball and catch it inside the cup.",
            "The cup must catch the ball attached by the string.",
            "Control the arm to fling the ball up into the cup.",
            "Toss the ball on the string and land it in the cup.",
            "The ball must be swung upward and caught in the cup.",
            "Use the arm to swing the ball and position the cup to catch.",
            "Catch the ball in the cup by swinging and timing correctly.",
            "The cup-and-ball system must achieve a successful catch.",
            "Swing the ball upward with momentum and catch it in the cup.",
            "The goal is catching the ball in the cup by swinging.",
        ],
    },
    ("pendulum", "swingup"): {
        "agent": "the pendulum",
        "task_aspect": "swing-up control",
        "goals": [
            "Swing the torque-limited pendulum up to vertical.",
            "The pendulum must be brought from hanging to upright.",
            "Pump energy into the pendulum to swing it to the top.",
            "Control the limited actuator to invert the pendulum.",
            "The pendulum should reach the upright vertical position.",
            "Use energy pumping to swing the pendulum up to vertical.",
            "Raise the pendulum from its resting position to upright.",
            "The torque-limited pendulum must reach the inverted pose.",
            "Swing the pendulum up by building oscillation amplitude.",
            "The goal is swinging the pendulum to its upright position.",
        ],
    },
    ("acrobot", "swingup"): {
        "agent": "the acrobot",
        "task_aspect": "underactuated swing-up",
        "goals": [
            "Swing up the two-link acrobot to the upright position.",
            "The acrobot must raise its tip to maximum height.",
            "Control the elbow joint to swing both links upright.",
            "Pump energy through the elbow to invert the acrobot.",
            "The two-link pendulum should reach the upright pose.",
            "Use the actuated elbow to swing the acrobot up.",
            "Bring both links of the acrobot to vertical position.",
            "The underactuated acrobot must achieve full inversion.",
            "Swing the acrobot up using only elbow joint control.",
            "The goal is swinging the acrobot to upright position.",
        ],
    },
    ("humanoid", "stand"): {
        "agent": "the humanoid",
        "task_aspect": "whole-body balance",
        "goals": [
            "Make the humanoid robot stand upright and balanced.",
            "The humanoid must maintain a standing upright position.",
            "Keep the humanoid balanced on its feet without falling.",
            "Stabilize the complex humanoid body in a standing pose.",
            "The humanoid should hold an erect balanced posture.",
            "Control all joints to keep the humanoid standing upright.",
            "Maintain whole-body balance with the humanoid robot.",
            "The humanoid must stay standing without collapsing.",
            "Keep the humanoid's center of mass over its feet.",
            "The goal is stable upright standing for the humanoid.",
        ],
    },
    ("humanoid", "walk"): {
        "agent": "the humanoid",
        "task_aspect": "bipedal walking",
        "goals": [
            "Make the humanoid robot walk forward.",
            "The humanoid must advance forward with a walking gait.",
            "Control the humanoid to walk forward on two legs.",
            "Drive the humanoid forward using coordinated bipedal steps.",
            "The humanoid should walk forward with natural arm swing.",
            "Achieve forward walking with the complex humanoid body.",
            "The humanoid must locomote forward by walking.",
            "Walk the humanoid forward as quickly as possible.",
            "Coordinate all joints for forward bipedal walking.",
            "The goal is forward walking for the humanoid robot.",
        ],
    },
    ("humanoid", "run"): {
        "agent": "the humanoid",
        "task_aspect": "high-speed bipedal running",
        "goals": [
            "Make the humanoid robot run forward at maximum speed.",
            "The humanoid must sprint forward as fast as possible.",
            "Control the humanoid for high-speed forward running.",
            "Drive the humanoid forward at top bipedal running speed.",
            "The humanoid should run forward at its fastest velocity.",
            "Achieve maximum forward running speed with the humanoid.",
            "The humanoid must sprint forward using full body coordination.",
            "Run the humanoid forward as rapidly as possible.",
            "Maximize the humanoid's forward running velocity.",
            "The goal is fastest forward running for the humanoid.",
        ],
    },
}


def main():
    result = {}

    for (domain, task), info in TASKS.items():
        key = f"{domain}_{task}"
        goals = info["goals"]
        agent = info["agent"]
        task_aspect = info["task_aspect"]
        assert len(goals) == 10, f"{key}: need 10 goals, got {len(goals)}"

        texts = []
        rng = random.Random(hash((domain, task)))

        for cat in IRRELEVANT_CATEGORIES:
            templates = list(IRRELEVANT[cat])
            rng.shuffle(templates)
            for i in range(10):
                goal = goals[i]
                irr = templates[i % len(templates)].format(
                    agent=agent, task_aspect=task_aspect
                )
                text = f"GOAL: {goal} TASK IRRELEVANT: {irr}"
                texts.append({
                    "category": cat,
                    "goal_index": i,
                    "text": text,
                })

        assert len(texts) == 100, f"{key}: {len(texts)}"
        max_words = max(len(t["text"].split()) for t in texts)

        result[key] = {
            "domain": domain,
            "task": task,
            "num_texts": 100,
            "texts": texts,
        }
        print(f"  {key}: 100 texts, max {max_words} words")

    output = {
        "_metadata": {
            "description": (
                "Each text combines a GOAL description with a TASK IRRELEVANT explanation. "
                "The GOAL describes what the agent should achieve. "
                "The TASK IRRELEVANT explains WHY a specific visual category is not useful "
                "for the task, helping the encoder learn to suppress irrelevant visual features. "
                "Format: 'GOAL: <goal> TASK IRRELEVANT: <why_category_is_irrelevant>'"
            ),
            "irrelevant_categories": {
                "background_color": "Why background/sky color is irrelevant to control",
                "floor_texture": "Why ground surface texture/pattern is irrelevant to control",
                "floor_color": "Why ground surface color is irrelevant to control",
                "agent_color": "Why agent body color is irrelevant to control",
                "agent_material": "Why agent surface material/finish is irrelevant to control",
                "lighting": "Why lighting direction/color/intensity is irrelevant to control",
                "joint_markers": "Why visual joint markers are irrelevant to control",
                "shadow": "Why shadow appearance is irrelevant to control",
                "camera": "Why camera viewpoint/angle is irrelevant to control",
                "atmosphere": "Why fog/particles/haze effects are irrelevant to control",
            },
            "structure": "10 irrelevant categories x 10 goal variations = 100 texts per task",
            "usage": "random.choice(tasks[key]['texts'])['text'] at each episode start",
            "num_tasks": len(result),
            "texts_per_task": 100,
        },
        "tasks": result,
    }

    out = "/home/claude/dmc_task_texts.json"
    with open(out, "w") as f:
        json.dump(output, f, indent=2)
    sz = os.path.getsize(out)
    print(f"\n{len(result)} tasks x 100 = {len(result)*100} total texts")
    print(f"Written to {out} ({sz/1024:.1f} KB)")


if __name__ == "__main__":
    main()
