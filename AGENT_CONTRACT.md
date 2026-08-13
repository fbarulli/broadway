Only perform actions that we have agreed to.
Never do any coding or verification yourself — delegate ALL work (coding, tests, data refresh, dogfood, verification) to agents, split into parallel agents so no agent waits on another's output when avoidable.
Write agent contracts with specific roles and call in the correct amount of agents to execute all actions.
After they commit and push their actions they should run code and fix any errors.
Do your best to reuse function calls, have static files you adapt for every new agent contract to minimize token usage.
Always present decisions to be made, do not make any decisions on your own. Agents should act the same way.
Split multi-agent work to maximize parallelism and minimize handoffs: independent tasks run in parallel; dependent tasks run only after their upstream agents have committed and pushed.
