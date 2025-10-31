import argparse
import json
import dspy
import litellm
from dspy.teleprompt import BootstrapFewShot
import os

LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")

BLOOM_GUIDANCE_PROMPT = """
You are a helpful tutor.

Use the following guidance to craft your explanation depending on the level of the question:

- Remembering: Help the student recall key facts or terms. Use mnemonics or memory strategies.
- Understanding: Guide the student to paraphrase the concept and walk through the reasoning that leads to the answer.
- Applying: Show how to use prior knowledge in a new situation. Walk through the application.
- Analyzing: Break down the concept and compare parts or relationships. Walk through how to think critically to reach the answer.
- Evaluating: Help the student justify or critique a position using logic or evidence.
- Creating: Guide the student in combining ideas to produce something new. Walk through the steps of the design process.
"""
def get_training_data():
    return [
        # REMEMBERING
        dspy.Example(
            tutor_prompt=build_unified_prompt(
                "What is the chemical symbol for sodium?",
                "Na"
            ),
            bloom_level="Remembering",
            bloom_reasoning="This question is about recalling a specific fact (a chemical symbol), which matches the Remembering level of Bloom’s taxonomy.",
            explanation="This is a Remembering-level question. To answer it, you need to recall the symbol for sodium from the periodic table. A helpful strategy is to remember that some elements use Latin roots. Sodium’s Latin name is 'Natrium', which is why its symbol is 'Na'. A quick mnemonic: 'Na – Not Apparent from English!' to help remember it doesn't match the English spelling."
        ).with_inputs("tutor_prompt"),

        # UNDERSTANDING
        dspy.Example(
            tutor_prompt=build_unified_prompt(
                "Explain why the heart has four chambers.",
                "The heart has four chambers to separate oxygen-rich and oxygen-poor blood, ensuring efficient circulation throughout the body.",
            ),
            bloom_level="Understanding",
            bloom_reasoning="The student needs to explain how the heart works, showing comprehension of structure and function, which aligns with the Understanding level.",
            explanation="This is an Understanding-level question. To answer it, try to explain in your own words what each chamber does. The right side of the heart handles blood returning from the body and sends it to the lungs to pick up oxygen. The left side receives the oxygenated blood and pumps it to the body. Walking through the process helps you understand how the heart keeps the oxygen-rich and oxygen-poor blood separate to work efficiently."
        ).with_inputs("tutor_prompt"),

        # APPLYING
        dspy.Example(
            tutor_prompt=build_unified_prompt(
                "Use the Pythagorean theorem to find the length of the hypotenuse in a right triangle with legs of 3 and 4 units.",
                "The hypotenuse is 5 units. Using a² + b² = c²: 3² + 4² = 9 + 16 = 25, so √25 = 5."
            ),
            bloom_level="Applying",
            bloom_reasoning="The question requires applying a known formula (Pythagorean theorem) to a specific case, which falls under the Applying level.",
            explanation="This is an Applying-level question. Start by recognizing that this is a right triangle, so the Pythagorean theorem applies: a² + b² = c². Use the values: a = 3 and b = 4. Compute 3² = 9 and 4² = 16, then add them: 9 + 16 = 25. Now take the square root of 25 to find the hypotenuse: √25 = 5. This shows how to use prior knowledge in a new situation."
        ).with_inputs("tutor_prompt"),

        # ANALYZING
        dspy.Example(
            tutor_prompt=build_unified_prompt(
                "Compare mitosis and meiosis in terms of genetic variation and number of resulting cells.",
                "Mitosis creates two identical cells; meiosis creates four genetically unique cells. Mitosis is for growth, meiosis for reproduction."
            ),
            bloom_level="Analyzing",
            bloom_reasoning="The question involves breaking down two processes and comparing them across multiple dimensions (variation, cell count), which is classic Analyzing.",
            explanation="This is an Analyzing-level question. Start by breaking down each process: mitosis results in two identical cells, and is used for growth and repair. Meiosis results in four cells with half the DNA, all genetically different, and is used for reproduction. Notice the patterns: mitosis keeps the DNA the same; meiosis increases variation. You're analyzing by comparing structure, outcome, and function to understand deeper relationships."
        ).with_inputs("tutor_prompt"),

        # EVALUATING
        dspy.Example(
            tutor_prompt=build_unified_prompt(
                "Which programming language would you recommend for beginners and why?",
                "Python, because its simple syntax makes it easy to learn, and it's widely used in various fields from web development to AI."
            ),
            bloom_level="Evaluating",
            bloom_reasoning="This requires making a justified recommendation with reasoning, which aligns with the Evaluating level.",
            explanation="This is an Evaluating-level question. To answer, you must assess programming languages based on clarity, ease of learning, and real-world use. Python stands out due to its readable syntax and broad application. For example, 'print(\"Hello\")' in Python is much simpler than in Java or C++, making this language suitable for beginners."
        ).with_inputs("tutor_prompt"),

        # CREATING
        dspy.Example(
            tutor_prompt=build_unified_prompt(
                "Design an experiment to test the effect of light on plant growth.",
                "Place identical plants under different light conditions (sunlight, LED, darkness) and measure growth over 2 weeks to compare results."
            ),
            bloom_level="Creating",
            bloom_reasoning="This question asks the student to design a novel experiment, combining ideas into a new structure — the essence of the Creating level.",
            explanation="This is a Creating-level question. Start by defining your goal — testing how light affects growth. Then combine your knowledge of experimental design: choose a variable (light type), control others (same plant species, water, soil), and decide on a measurable outcome (growth in cm). By generating this plan, you're combining ideas to build a new solution from scratch — the hallmark of creative thinking."
        ).with_inputs("tutor_prompt")
    ]


# Signature 
class FullChainSignature(dspy.Signature):
    tutor_prompt = dspy.InputField()
    bloom_level = dspy.OutputField(desc="Predicted Bloom level")
    bloom_reasoning = dspy.OutputField(desc="Why this is that Bloom level")
    explanation = dspy.OutputField(desc="Tutor-style explanation with reasoning and strategy")

# Module 
class FullChainExplanation(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(FullChainSignature)

    def forward(self, tutor_prompt):
        return self.generate(tutor_prompt=tutor_prompt)

def build_unified_prompt(question, answer):
    return (
        f"{BLOOM_GUIDANCE_PROMPT.strip()}\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
    )

def compile_model():
    examples = get_training_data()
    booster = BootstrapFewShot(metric=lambda e, p, _: 1, max_bootstrapped_demos=6)
    return booster.compile(FullChainExplanation(), trainset=examples)

# Main
def main():
    parser = argparse.ArgumentParser(description="Predict Bloom level + generate explanation in one chain")
    parser.add_argument("--input", type=str, default="bloom_qa_elementary_math.json")
    parser.add_argument("--output", type=str, default="test.json")
    args = parser.parse_args()

    lm = dspy.LM(
        model='openai/gpt-4o-mini',
        api_key=LITELLM_API_KEY,
        api_base=LITELLM_BASE_URL,
        temperature=0.7,
        top_p=0.9
    )
    dspy.configure(lm=lm)

    
    with open(args.input, "r") as f:
        data = json.load(f)
    model = compile_model()
    # Run inference
    count = 0
    for item in data:
        question = item["question"]
        answer = item["answer"]
        prompt = build_unified_prompt(question, answer)

        result = model(tutor_prompt=prompt)

        item["predicted_bloom_level"] = result.bloom_level.strip()
        item["bloom_reasoning"] = result.bloom_reasoning.strip()
        item["explanation"] = result.explanation.strip()
        count += 1

    # Save output
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nGenerated explanations for {count} examples.")
    print(f"Output saved to: {args.output}")
if __name__ == "__main__":
    main()
