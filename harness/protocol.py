"""The settings both conditions share.

The paper's state machine allowed 40 rounds per model step. Agentomics-ML 1.2.0's agents do more per step and ran out of
40 rounds in its exploration and representation steps, so both conditions use its own default of 100 instead, and the
skill's budget follows the paper's rule: the per-step budget times five model steps times three iterations.
"""

ITERATIONS = 3
ROUNDS_PER_STEP = 100     # state machine: rounds (a model call and the tool calls it asks for) per model step
RETRIES = 3               # state machine: returns to the step that produced a failing script
SKILL_ROUNDS = ROUNDS_PER_STEP * 5 * ITERATIONS   # skill: rounds in its one turn
TEMPERATURE = 0.7         # Agentomics-ML's default, used for the skill agent too
