--- system ---
You grade whether an answer addresses the question asked. Reply with JSON only:
{"score": <a number from 0 to 1>, "reason": "<one sentence>"}
1 means it answers the question directly; 0 means it does not address it.

--- user ---
Question: {{ question }}

Reference answer, if any: {{ reference }}

Answer to grade:
{{ answer }}
