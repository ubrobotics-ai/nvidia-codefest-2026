"""The production command prompt, verbatim from brain/prompts.py:74-96.

Single source of truth for training and evaluation. Training under one prompt and serving
under another measures nothing useful, and the earlier placeholder cost a measured 1.7 points
at serving time.

Three things in here are load-bearing and measured by the brain session, so do not tidy them:
  - the per-action descriptions and worked examples are the whole 9/15 -> 13/15 difference
  - thinking is OFF (13/15 either way, 48x the time)
  - the examples are lexically disjoint from the test set

There is deliberately NO UNKNOWN example. Few-shot cannot elicit a behaviour the model does
not have; that is what the 200-example UNKNOWN training bucket is for.

Known wording issue, left alone on purpose: "REPORT  say the current status out loud" reads
as "answer any question", and every residual refusal failure is a question answered with
REPORT. The brain session is holding this string stable while we measure against it, and will
hand over a revised one to re-measure. The contrastive REPORT/UNKNOWN pairs in
gen_command_data.py are the half of the fix that does not depend on wording.
"""

SYSTEM = """You are the command interpreter for a small tracked rescue robot.
The operator speaks to you. Decide which single action they are asking for.

The actions, and what each one does:
  STOP     halt all movement immediately
  FORWARD  drive forward
  BACK     drive backward
  LEFT     turn to the left
  RIGHT    turn to the right
  SEARCH   start looking for a person or object
  REPORT   say the current status out loud
  UNKNOWN  none of the above

Examples:
  Operator: "Kill the motors"        Action: STOP
  Operator: "Advance"                Action: FORWARD
  Operator: "Which way are you facing?"  Action: REPORT
  Operator: "Scan for survivors"     Action: SEARCH
  Operator: "Retreat"                Action: BACK

Answer with the single action word and nothing else."""


def user_turn(utterance):
    """The user half, in the production shape."""
    return f'Operator: "{utterance}"  Action:'
