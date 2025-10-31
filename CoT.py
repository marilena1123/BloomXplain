import json
import re
import argparse
from openai import OpenAI
import os

LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")

parser = argparse.ArgumentParser(description="Run CoT")
parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
parser.add_argument("--output_file", type=str, required=True, help="Path to save the output JSON file.")
parser.add_argument("--task", type=str, required=True, help="Task name (e.g., 'gsm', 'snarks', 'object_counting').")

args = parser.parse_args()

INPUT_FILE = args.input_file
OUTPUT_FILE = args.output_file
TASK = args.task


if TASK=="gsm":
    SYSTEM_PROMPT="You are an expert in math reasoning. Please solve the math word problems step by step. At the end of your answer write 'The final answer is X'."
elif TASK=="snarks":
    SYSTEM_PROMPT="Determine which of two sentences is sarcastic. According to Cambridge University Dictionary, sarcasm is 'the use of remarks that clearly mean the opposite of what they say, made in order to hurt someone's feelings or to criticize something in a humorous way.' Sarcastic sentences often contain satirical or ironic utterances, hyperboles, ambivalent or witty remarks. Answer the question step-by-step. At the end of your answer, write: 'The final answer is X'."
elif TASK=="object_counting":
    SYSTEM_PROMPT="Think step-by-step. At the end of your answer write 'The final answer is X'."
else:
    SYSTEM_PROMPT="Clarify the meaning of sentences with ambiguous pronounces. Think step-by-step. At the end of your answer write 'The final answer is X'."

FEW_SHOT_EXAMPLES = {
    "object_counting": [
        {
            "role": "user",
            "content": "I have a blackberry, a clarinet, a nectarine, a plum, a strawberry, a banana, a flute, an orange, and a violin. How many fruits do I have?"
        },
        {
            "role": "assistant",
            "content": """Let's think step by step.
We first identify the fruits on the list and include their quantity in parentheses:
- blackberry (1)
- nectarine (1)
- plum (1)
- strawberry (1)
- banana (1)
- orange (1)
Now, let's add the numbers in parentheses: 1 + 1 + 1 + 1 + 1 + 1 = 6.
The answer is 6."""
        },
        {
            "role": "user",
            "content": "I have an orange, a raspberry, two peaches, a blackberry, an apple, a grape, a nectarine, and three plums. How many fruits do I have?"
        },
        {
            "role": "assistant",
            "content": """Let's think step by step.
We first identify the fruits on the list and include their quantity in parentheses:
- orange (1)
- raspberry (1)
- peaches (2)
- blackberry (1)
- apple (1)
- grape (1)
- nectarine (1)
- plums (3)
Now, let's add the numbers in parentheses: 1 + 1 + 2 + 1 + 1 + 1 + 1 + 3 = 11.
The answer is 11."""
        },
        {
            "role": "user",
            "content": "I have a lettuce head, a head of broccoli, an onion, a stalk of celery, two carrots, a garlic, and a yam. How many vegetables do I have?"
        },
        {
            "role": "assistant",
            "content": """Let's think step by step.
We first identify the vegetables on the list and include their quantity in parentheses:
- lettuce (1)
- broccoli (1)
- onion (1)
- celery (1)
- carrots (2)
- garlic (1)
- yam (1)
Now, let's add the numbers in parentheses: 1 + 1 + 1 + 1 + 2 + 1 + 1 = 8.
The answer is 8."""
        }
    ],

    "disambiguation": [
        {
            "role": "user",
            "content": "In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\nSentence: The chief told the counselor that they took the day off.\nOptions:\n(A) The chief took the day off\n(B) The counselor took the day off\n(C) Ambiguous"
        },
        {
            "role": "assistant",
            "content": "Let’s think step by step.\nHere we need to determine who the pronoun “they” might be referring to. There are two possible referents for “they”, namely the chief and the counselor. The verb “told” might be able to help us determine which one is more likely (if either). Let X be the chief and Y the counselor. The sentence is then of the form “X told Y that (X or Y) did something.”\nLet’s consider Y first: “X told Y that Y did something.” This case does not make much sense, as Y would already have the information that Y did something, because it is information about themself.\nNow, consider X: “X told Y that X did something.” This makes sense, because X would be sharing some information about themself that Y might not have known before.\nBecause in this context, X is the chief and Y is the counselor, the answer should be the chief. So the answer is (A)."
        },
        {
            "role": "user",
            "content": "In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\nSentence: The manager sent a message to the secretary, but he didn’t reply yet.\nOptions:\n(A) The secretary didn’t reply yet\n(B) The manager didn’t reply yet\n(C) Ambiguous"
        },
        {
            "role": "assistant",
            "content": "Let’s think step by step.\nHere we need to determine who the pronoun “he” might be referring to. There are two possible referents for “he”, namely the manager and the secretary. The verbs “sent” and “reply” might be able to help us determine which one is more likely (if either). Let X be the manager and Y the secretary. The sentence is then of the form “X sent a message to Y, but (X or Y) didn’t reply yet.”\nLet’s consider Y first: “X sent a message to Y, but Y didn’t reply yet.” This case makes sense, because of the implicit causality of the sentence. Y was the receiver of the message, but Y didn’t get back to X yet.\nNow, consider X: “X sent a message to Y, but X didn’t reply yet.” This case doesn’t make sense, because X was the initial sender of the message, so it is now Y’s turn to write back to X.\nBecause in this context, X is the manager and Y is the secretary, the answer should be the secretary. So the answer is (A)."
        },
        {
            "role": "user",
            "content": "In the following sentences, explain the antecedent of the pronoun (which thing the pronoun refers to), or state that it is ambiguous.\nSentence: Bailey will plan to meet the director at his office\nOptions:\n(A) It will be Bailey’s office\n(B) It will be the director’s office\n(C) Ambiguous"
        },
        {
            "role": "assistant",
            "content": "Let’s think step by step.\nHere we need to determine who the pronoun “his” might be referring to. There are two possible referents for “his”, namely Bailey’s and the director’s. The verb phrase “plan to meet” might be able to help us determine which one is more likely (if either). Let X be Bailey and Y the director. The sentence is then of the form “X will plan to meet Y at (X or Y)’s office.”\nLet’s consider Y first: “X will plan to meet Y at Y’s office.” This case makes sense, because X might want to meet up with Y at Y’s own office.\nNow, consider X: “X will plan to meet Y at X’s office.” This case also makes sense, because X might want to meet up with Y at X’s own office.\nBecause both X and Y are possible at the same time, we conclude that the antecedent of the pronoun is ambiguous. So the answer is (C)."
        }
    ],
    
    "snarks": [
    {
        "role": "user",
        "content": "Which statement is sarcastic?\nOptions:\n(A) Yes, because having interests and actively researching them is a huge waste\n(B) Yes, because having interests and actively researching them is a huge deal"
    },
    {
        "role": "assistant",
        "content": "Let's think step by step.\nIf we look at (A), it says that having interests and actively researching them is a huge waste, implying that it is a useless effort.\nHowever, we know thahaving interests and actively researching them is typically not a waste but rather is beneficial to the individual. The presence of such a juxtaposition in (A) suggests that it contains a taste of irony  nd sarcasm.\nIf we look at (B), it says that having interests and actively researching them is a huge deal, implying that it is an important and consequential effort. This is arguably a neutral and correct statement.\nAbove the above, the sarcastic option is (A). The final answer is (A)."
    },
    {
        "role": "user",
        "content": "Which statement is sarcastic?\nOptions:\n(A) No one is going to disagree with you on this. Avoiding ad hominem attacks really help your case\n(B) No one is going to disagree with you on this. Ad hominem attacks really help your case"
    },
    {
        "role": "assistant",
        "content": "Let's think step by step.\nIf we look at (A), it says that avoiding ad hominem attacks really help your case, implying that ad hominem attacks are adverse and injurious. Because ad hominem attacks are addressed at a person rather than an idea, it is indeed true that avoiding them is often useful and helpful; so, (A) is a neutral (valid and agreeable) statement.\nIf we look at (B), isays that ad hominem attacks really help your case, implying that ad hominem attacks are a positive thing. However, we stated previously that ad hominem attacks are often not useful or constructive. The speaker in this sentence therefore seems to mean the opposite of what they are saying; so, there appears to have a taste of irony and sarcasm in (B).\nAbove the above, the sarcastic option is (B). The final answer is (B)."
    },
    {
        "role": "user",
        "content": "Which statement is sarcastic?\nOptions:\n(A) Consistency in the league's punishments? What do you think this is supposed to be, politics?\n(B) Consistency in the league's punishments? What do you think this is supposed to be, moral?"
    },
    {
        "role": "assistant",
        "content": "Let's think step by step.\nIf we look at (A), it likens the consistency in the league's punishments with that in politics. Because politics or political affairs are often not considered to be consistent or dependable, this sentence appears to be satirical.\nIf we look at (B), it likens the consistency in the league's punishments with that in morality. Discussing the consistency of t e league's unishments in the context of morality, ethics, or law makes sense and does not appear to make a satirical point about anything.\nAbove the above, the sarcastic option is (A). The final answer is (A)."
    }
    ],
    "gsm": [
    {
        "role": "user",
        "content": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"
    },
    {
        "role": "assistant",
        "content": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The final answer is 8."
    },
    {
        "role": "user",
        "content": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?"
    },
    {
        "role": "assistant",
        "content": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The final answer is 33."
    },
    {
        "role": "user",
        "content": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?"
    },
    {
        "role": "assistant",
        "content": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The final answer is 29."
    }
    ]
}

if TASK not in FEW_SHOT_EXAMPLES:
    raise ValueError(f"Unknown task: {TASK}. Please define few-shot examples for it.")

FEW_SHOT_PROMPT = FEW_SHOT_EXAMPLES[TASK]


client = OpenAI(api_key=LITELLM_API_KEY, base_url=LITELLM_BASE_URL)


with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

results = []


# === Inference loop ===
for sample in data:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + FEW_SHOT_PROMPT + [
        {"role": "user", "content": sample["input"]}
    ]

    response = client.chat.completions.create(
        model='openai/gpt-4o-mini',
        messages=messages,
    )

    full_response = response.choices[0].message.content.strip()

    results.append({
        "input": sample["input"],
        "target": sample["target"],
        "response": full_response
    })


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump({"results": results}, f, ensure_ascii=False, indent=2)

print(f"Benchmark complete")
