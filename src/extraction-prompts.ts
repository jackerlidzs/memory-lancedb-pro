/**
 * Prompt templates for intelligent memory extraction.
 * Three mandatory prompts:
 * - buildExtractionPrompt: 6-category L0/L1/L2 extraction with few-shot
 * - buildDedupPrompt: CREATE/MERGE/SKIP dedup decision
 * - buildMergePrompt: Memory merge with three-level structure
 */

export function buildExtractionPrompt(
  conversationText: string,
  user: string,
): string {
  return `Analyze the following session context and extract memories worth long-term preservation.

User: ${user}

Target Output Language: auto (detect from recent messages)

## Recent Conversation
${conversationText}

# Memory Extraction Criteria

## What is worth remembering?
- Personalized information: Information specific to this user, not general domain knowledge
- Long-term validity: Information that will still be useful in future sessions
- Specific and clear: Has concrete details, not vague generalizations

## What is NOT worth remembering?
- General knowledge that anyone would know
- Temporary information: One-time questions or conversations
- Vague information: "User has questions about a feature" (no specific details)
- Tool output, error logs, or boilerplate
- Recall queries / meta-questions: "Do you remember X?", "你还记得X吗?", "你知道我喜欢什么吗" — these are retrieval requests, NOT new information to store
- Degraded or incomplete references: If the user mentions something vaguely ("that thing I said"), do NOT invent details or create a hollow memory
- System messages: Model switch notifications, session timestamps, heartbeat messages
- Internal prompts: Memory flush instructions, compaction prompts, "reply with NO_REPLY"
- Raw system output: "[timestamp] Model switched to X" is NOT user knowledge

# Memory Classification

## Core Decision Logic

| Question | Answer | Category |
|----------|--------|----------|
| Who is the user? | Identity, attributes | profile |
| What does the user prefer? | Preferences, habits | preferences |
| What is this thing? | Person, project, organization | entities |
| What happened? | Decision, milestone | events |
| How was it solved? | Problem + solution | cases |
| What is the process? | Reusable steps | patterns |

## Precise Definition

**profile** - User identity (static attributes). Test: "User is..."
**preferences** - User preferences (tendencies). Test: "User prefers/likes..."
**entities** - Continuously existing nouns. Test: "XXX's state is..."
**events** - Things that happened. Test: "XXX did/completed..."
**cases** - Problem + solution pairs. Test: Contains "problem -> solution"
**patterns** - Reusable processes. Test: Can be used in "similar situations"

## Common Confusion
- "Plan to do X" -> events (action, not entity)
- "Project X status: Y" -> entities (describes entity)
- "User prefers X" -> preferences (not profile)
- "Encountered problem A, used solution B" -> cases (not events)
- "General process for handling certain problems" -> patterns (not cases)

# Three-Level Structure

Each memory contains three levels. CRITICAL: Each level MUST contain DIFFERENT content. NEVER copy the same text across levels.

**abstract (L0)**: One-liner index (MAX 15 words). This is a short label/key, NOT a full sentence.
- Merge types (preferences/entities/profile/patterns): \`[Merge key]: [Description]\`
- Independent types (events/cases): Short specific label

**overview (L1)**: Structured Markdown summary with bullet points and category-specific headings. Must use markdown formatting (##, -, bullet lists). Must be MORE structured than L0 and SHORTER than L2.

**content (L2)**: Full narrative paragraph with background context, reasoning, and complete details. Must be a proper sentence/paragraph, NOT a copy of L0.

# Few-shot Examples

## profile — GOOD example (L0 ≠ L1 ≠ L2)
\`\`\`json
{
  "category": "profile",
  "abstract": "User: AI engineer, 3 years LLM experience",
  "overview": "## Background\\n- Occupation: AI development engineer\\n- Experience: 3 years LLM development\\n- Tech stack: Python, LangChain",
  "content": "The user is a professional AI development engineer who has been working with Large Language Model applications for 3 years. Their primary tech stack includes Python and LangChain for building LLM-powered systems."
}
\`\`\`

## preferences — GOOD example (L0 ≠ L1 ≠ L2)
\`\`\`json
{
  "category": "preferences",
  "abstract": "Code style: no type hints, concise",
  "overview": "## Preference Domain\\n- Language: Python\\n- Topic: Code style\\n\\n## Details\\n- No type hints\\n- Concise function comments\\n- Direct implementation over abstraction",
  "content": "The user prefers writing Python code without type hints, favoring concise function comments and direct implementation patterns. They value readability through simplicity rather than verbose type annotations."
}
\`\`\`

## cases — GOOD example (L0 ≠ L1 ≠ L2)
\`\`\`json
{
  "category": "cases",
  "abstract": "LanceDB BigInt error → Number() coercion fix",
  "overview": "## Problem\\nLanceDB 0.26+ returns BigInt for numeric columns, causing arithmetic errors\\n\\n## Solution\\nWrap values with Number() before arithmetic operations",
  "content": "When upgrading to LanceDB 0.26+, numeric columns like timestamp and importance are returned as BigInt instead of Number. This causes TypeErrors when performing arithmetic. The fix is to coerce values with Number(value) before any math operations."
}
\`\`\`

## BAD example — DO NOT do this (L0 = L1 = L2)
\`\`\`json
{
  "abstract": "User prefers Python for AI development",
  "overview": "User prefers Python for AI development",
  "content": "User prefers Python for AI development"
}
\`\`\`
This is WRONG because all three levels are identical copies. Each level must provide different granularity of information.

# Output Format

Return JSON:
{
  "memories": [
    {
      "category": "profile|preferences|entities|events|cases|patterns",
      "abstract": "One-line index",
      "overview": "Structured Markdown summary",
      "content": "Full narrative"
    }
  ]
}

Notes:
- Output language should match the dominant language in the conversation
- Only extract truly valuable personalized information
- If nothing worth recording, return {"memories": []}
- Maximum 5 memories per extraction
- Preferences should be aggregated by topic`;
}

export function buildDedupPrompt(
  candidateAbstract: string,
  candidateOverview: string,
  candidateContent: string,
  existingMemories: string,
): string {
  return `Determine how to handle this candidate memory.

**Candidate Memory**:
Abstract: ${candidateAbstract}
Overview: ${candidateOverview}
Content: ${candidateContent}

**Existing Similar Memories**:
${existingMemories}

Please decide:
- SKIP: Candidate memory duplicates existing memories, no need to save. Also SKIP if the candidate contains LESS information than an existing memory on the same topic (information degradation — e.g., candidate says "programming language preference" but existing memory already says "programming language preference: Python, TypeScript")
- CREATE: This is completely new information not covered by any existing memory, should be created
- MERGE: Candidate memory adds genuinely NEW details to an existing memory and should be merged

IMPORTANT:
- "events" and "cases" categories are independent records — they do NOT support MERGE. For these categories, only use SKIP or CREATE.
- If the candidate appears to be derived from a recall question (e.g., "Do you remember X?" / "你记得X吗？") and an existing memory already covers topic X with equal or more detail, you MUST choose SKIP.
- A candidate with less information than an existing memory on the same topic should NEVER be CREATED or MERGED — always SKIP.

Return JSON format:
{
  "decision": "skip|create|merge",
  "match_index": 1,
  "reason": "Decision reason"
}

If decision is "merge", set "match_index" to the number of the existing memory to merge with (1-based).`;
}

export function buildMergePrompt(
  existingAbstract: string,
  existingOverview: string,
  existingContent: string,
  newAbstract: string,
  newOverview: string,
  newContent: string,
  category: string,
): string {
  return `Merge the following memory into a single coherent record with all three levels.

**Category**: ${category}

**Existing Memory:**
Abstract: ${existingAbstract}
Overview:
${existingOverview}
Content:
${existingContent}

**New Information:**
Abstract: ${newAbstract}
Overview:
${newOverview}
Content:
${newContent}

Requirements:
- Remove duplicate information
- Keep the most up-to-date details
- Maintain a coherent narrative
- Keep code identifiers / URIs / model names unchanged when they are proper nouns

Return JSON:
{
  "abstract": "Merged one-line abstract",
  "overview": "Merged structured Markdown overview",
  "content": "Merged full content"
}`;
}
