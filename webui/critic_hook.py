"""Claude Code PreToolUse hook: a review is filed by a sub-agent (the critic), never by the main
conversation.

The web interface refuses the working agent's submit_critic_review on its own agent path, where
every call passes through it. A Claude Code run talks to openPASO directly, so the refusal is
made here, in the hook Claude Code runs before each call of the filing tool (Copilot, 2026-10-01:
the only-critic rule did not reach Claude Code runs). Claude Code puts `agent_id` into a hook's
input only when the call comes from inside a sub-agent (its hook input schema, checked in the
installed 2.1.281: "Subagent identifier. Present only when the hook fires inside a subagent").

Claude Code cannot tell a critic from another kind of sub-agent, so on this path any sub-agent
may file; the main conversation may not.
"""
import json
import sys

REFUSAL = ("In this interface only the critic files a review. A review of your own work filed by "
           "you is not a review, so this call was not passed to openPASO. Spawn a critic sub-agent, "
           "give it the file to review by its name in the run folder, and let it file its own "
           "verdict; then run that file unchanged with input_path (run_simulation) or "
           "generator_path (run_with_generator).")


def decide(event: dict) -> dict | None:
    """The hook's answer: a refusal for the main conversation, nothing for a sub-agent."""
    if event.get("agent_id"):
        return None
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                   "permissionDecision": "deny",
                                   "permissionDecisionReason": REFUSAL}}


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except ValueError:
        event = {}
    answer = decide(event if isinstance(event, dict) else {})
    if answer is not None:
        print(json.dumps(answer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
