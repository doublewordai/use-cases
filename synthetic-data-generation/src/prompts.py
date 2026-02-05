"""Prompts and JSON schemas for synthetic data generation.

Uses OpenAI-compatible structured outputs to guarantee valid JSON responses.
See: https://platform.openai.com/docs/guides/structured-outputs
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
    "Each scenario should include a realistic customer name, the topic, "
    "difficulty level, a specific situation (2-3 sentences describing the problem), "
    "what the customer already tried, their sentiment, and desired number of turns."
)

SCENARIO_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "scenario",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "customer_name": {
                    "type": "string",
                    "description": "A realistic customer name",
                },
                "topic": {
                    "type": "string",
                    "description": "The support topic category",
                },
                "difficulty": {
                    "type": "string",
                    "description": "Difficulty level of the scenario",
                    "enum": ["easy", "medium", "hard"],
                },
                "situation": {
                    "type": "string",
                    "description": "2-3 sentences describing the specific problem",
                },
                "prior_attempts": {
                    "type": "string",
                    "description": "What the customer already tried",
                },
                "sentiment": {
                    "type": "string",
                    "description": "Customer's emotional state",
                    "enum": ["frustrated", "neutral", "positive"],
                },
                "desired_turns": {
                    "type": "integer",
                    "description": "Target number of conversation turns (3-8)",
                },
            },
            "required": [
                "customer_name",
                "topic",
                "difficulty",
                "situation",
                "prior_attempts",
                "sentiment",
                "desired_turns",
            ],
            "additionalProperties": False,
        },
    },
}

# --- Conversation Stage ---

CONVERSATION_SYSTEM_PROMPT = (
    "You are generating training conversations for a {domain} AI. "
    "Given a scenario, generate a natural multi-turn conversation between a "
    "Customer and an Agent. The agent should be helpful, empathetic, and "
    "follow best practices: acknowledge the issue, ask clarifying questions, "
    "provide step-by-step solutions, and confirm resolution."
)

CONVERSATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "conversation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "description": "The conversation messages",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {
                                "type": "string",
                                "description": "Who is speaking",
                                "enum": ["customer", "agent"],
                            },
                            "content": {
                                "type": "string",
                                "description": "The message content",
                            },
                        },
                        "required": ["role", "content"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["messages"],
            "additionalProperties": False,
        },
    },
}

# --- Quality Stage ---

QUALITY_SYSTEM_PROMPT = (
    "You are a quality evaluator for customer support conversations. Score "
    "the following conversation on three dimensions (1-5 each): naturalness "
    "(does it sound like a real conversation?), helpfulness (does the agent "
    "resolve the issue effectively?), guidelines (does the agent follow "
    "support best practices?). Provide brief reasoning for your scores."
)

QUALITY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "quality_score",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "naturalness": {
                    "type": "integer",
                    "description": "Score 1-5 for how natural the conversation sounds",
                },
                "helpfulness": {
                    "type": "integer",
                    "description": "Score 1-5 for how effectively the agent resolves the issue",
                },
                "guidelines": {
                    "type": "integer",
                    "description": "Score 1-5 for adherence to support best practices",
                },
                "overall": {
                    "type": "number",
                    "description": "Average of the three scores",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of the scores",
                },
            },
            "required": [
                "naturalness",
                "helpfulness",
                "guidelines",
                "overall",
                "reasoning",
            ],
            "additionalProperties": False,
        },
    },
}
