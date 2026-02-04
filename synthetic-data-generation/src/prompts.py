"""Prompts and configuration for synthetic data generation."""

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

SCENARIO_SYSTEM_PROMPT = (
    "You are a scenario designer for {domain} training data. "
    "Generate realistic {domain} scenarios for a {product}. "
    "Each scenario should include: customer_name (realistic name), topic, "
    "difficulty, situation (2-3 sentences describing the specific problem), "
    "prior_attempts (what the customer already tried), sentiment "
    "(frustrated/neutral/positive), desired_turns (3-8 based on difficulty). "
    "Return valid JSON."
)

CONVERSATION_SYSTEM_PROMPT = (
    "You are generating training conversations for a {domain} AI. "
    "Given a scenario, generate a natural multi-turn conversation between a "
    "Customer and an Agent. The agent should be helpful, empathetic, and "
    "follow best practices: acknowledge the issue, ask clarifying questions, "
    "provide step-by-step solutions, and confirm resolution. Return a JSON "
    "object with a 'messages' array where each message has 'role' "
    "(customer/agent) and 'content'."
)

QUALITY_SYSTEM_PROMPT = (
    "You are a quality evaluator for customer support conversations. Score "
    "the following conversation on three dimensions (1-5 each): naturalness "
    "(does it sound like a real conversation?), helpfulness (does the agent "
    "resolve the issue effectively?), guidelines (does the agent follow "
    "support best practices?). Return JSON with: naturalness, helpfulness, "
    "guidelines, overall (average of three), reasoning."
)
