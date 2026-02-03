"""Prompt templates for the async research agent with web search."""

QUERY_GENERATION_SYSTEM = (
    "You are a research planning assistant. Your job is to generate effective "
    "web search queries that will find relevant, authoritative information."
)

SEED_QUERY_PROMPT = (
    "Generate {count} diverse web search queries to research the following topic. "
    "The queries should cover different angles and aspects of the topic. Use "
    "specific, targeted search terms that will return high-quality results.\n\n"
    "Topic: {topic}\n\n"
    "Return exactly {count} queries, numbered 1 through {count}. Each query "
    "should be a concise search string (not a question), designed to find "
    "authoritative sources."
)

FOLLOWUP_QUERY_PROMPT = (
    "Based on the research findings so far, generate {count} follow-up web search "
    "queries to fill gaps and deepen understanding.\n\n"
    "Topic: {topic}\n\n"
    "Findings so far:\n{findings}\n\n"
    "Generate {count} follow-up search queries that:\n"
    "1. Address gaps or uncertainties in the current findings\n"
    "2. Explore connections between different findings\n"
    "3. Go deeper on the most important results\n"
    "4. Cover angles not yet explored\n\n"
    "Return exactly {count} queries, numbered 1 through {count}. Each query "
    "should be a concise search string designed to find new information."
)

ANALYSIS_SYSTEM = (
    "You are a research analyst. Given web page content and a research topic, "
    "extract and summarize the key information relevant to the topic. Be specific "
    "about facts, figures, dates, and claims. Note the credibility and nature of "
    "the source. If the content is not relevant, say so briefly."
)

ANALYSIS_PROMPT = (
    "Research topic: {topic}\n\n"
    "Source URL: {url}\n"
    "Source title: {title}\n\n"
    "Web page content:\n{content}\n\n"
    "Extract and summarize the key information from this source that is relevant "
    "to the research topic. Include specific facts, figures, and claims. Note the "
    "nature and apparent credibility of this source."
)

SYNTHESIS_SYSTEM = (
    "You are a research report writer. Given a collection of analyzed web sources "
    "on a topic, synthesize them into a coherent, well-structured report. Organize "
    "by themes, not by source. Include an executive summary, key findings, source "
    "citations, and areas where sources disagree or information is uncertain."
)

SYNTHESIS_PROMPT = (
    "Topic: {topic}\n\n"
    "Analyzed sources from {num_rounds} rounds of web research:\n\n"
    "{findings}\n\n"
    "Synthesize these findings into a comprehensive research report. Include:\n"
    "1. Executive summary\n"
    "2. Key findings organized by theme\n"
    "3. Areas of agreement and disagreement between sources\n"
    "4. Gaps and areas for further research\n"
    "5. Source list with URLs"
)
