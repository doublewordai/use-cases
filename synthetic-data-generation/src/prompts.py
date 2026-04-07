"""Prompts and JSON schemas for synthetic data generation.

Uses response_format=json_object for broad model compatibility. The expected
JSON structure is described in the system prompt so the model knows what
fields to produce.
"""

SUPPORT_TOPICS = [
    "Billing & Payments",
    "Account Access",
    "API Integration",
    "Performance Issues",
    "Data Export",
    "Team Management",
    "Security & Privacy",
    "Feature Requests",
    "Onboarding",
    "Downtime & Outages",
    "Subscription Changes",
    "Compliance",
    "Mobile App",
    "Webhooks & Notifications",
    "Documentation",
]

DIFFICULTY_DISTRIBUTION = {
    "easy": 0.40,
    "medium": 0.35,
    "hard": 0.25,
}

# --- Scenario Stage ---

SCENARIO_SYSTEM_PROMPT = (
    "You are a scenario designer for {domain} training data. "
    "Generate realistic {domain} scenarios for a {product}. "
    "Respond with a JSON object containing exactly these fields:\n"
    '- "customer_name": string (a realistic name)\n'
    '- "topic": string (the support topic category)\n'
    '- "difficulty": string (one of: "easy", "medium", "hard")\n'
    '- "situation": string (2-3 sentences describing the specific problem)\n'
    '- "prior_attempts": string (what the customer already tried)\n'
    '- "sentiment": string (one of: "frustrated", "neutral", "positive")\n'
    '- "desired_turns": integer (target conversation turns, 3-8)\n'
    "Output only valid JSON, no other text."
)

SCENARIO_SCHEMA = {"type": "json_object"}

# --- Conversation Stage ---

CONVERSATION_SYSTEM_PROMPT = (
    "You are generating training conversations for a {domain} AI. "
    "Given a scenario, generate a natural multi-turn conversation between a "
    "Customer and an Agent. The agent should be helpful, empathetic, and "
    "follow best practices: acknowledge the issue, ask clarifying questions, "
    "provide step-by-step solutions, and confirm resolution.\n"
    "Respond with a JSON object containing exactly one field:\n"
    '- "messages": array of objects, each with "role" (either "customer" or "agent") '
    'and "content" (the message text)\n'
    "Output only valid JSON, no other text."
)

CONVERSATION_SCHEMA = {"type": "json_object"}

# --- Quality Stage ---

QUALITY_SYSTEM_PROMPT = (
    "You are a quality evaluator for customer support conversations. Score "
    "the following conversation on three dimensions (1-5 each): naturalness "
    "(does it sound like a real conversation?), helpfulness (does the agent "
    "resolve the issue effectively?), guidelines (does the agent follow "
    "support best practices?). Provide brief reasoning for your scores.\n"
    "Respond with a JSON object containing exactly these fields:\n"
    '- "naturalness": integer (1-5)\n'
    '- "helpfulness": integer (1-5)\n'
    '- "guidelines": integer (1-5)\n'
    '- "overall": number (average of the three scores)\n'
    '- "reasoning": string (brief explanation)\n'
    "Output only valid JSON, no other text."
)

QUALITY_SCHEMA = {"type": "json_object"}
