import re

def deterministic_guardials(text: str) -> bool:
    keyword = ["murder", "suicide", "kill", "bomb"]
    return any(kw in text.lower() for kw in keyword)

test_inputs = [
    # Normal
    "Hello, how are you?",
    "What's the weather today?",
    "I want to learn Python programming.",
    "Can you recommend a good movie?",
    "Let's build a chatbot using LangChain.",

    # Should be blocked
    "I want to kill someone.",
    "How do I make a bomb?",
    "This movie is about a murder mystery.",
    "He committed suicide yesterday.",
    "KILL them all!",

    # Edge cases
    "The Bombastic soundtrack is amazing.",   # Contains "bomb" as part of another word
    "Murder mysteries are my favorite genre.",
    "The killer whale is a fascinating animal.",  # Doesn't contain exact "kill"
    "Suicidal thoughts should be discussed with a professional.",  # Doesn't contain exact "suicide"
    "Skill development is important.",  # Contains "kill" inside "skill"
]

for inp in test_inputs:
    blocked = deterministic_guardials(inp)
    status = "BLOCKED" if blocked else "NORMAL"
    print(f"{status:8} | {inp}")