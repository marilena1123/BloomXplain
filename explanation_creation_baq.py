import argparse
import json
import dspy
from dspy.teleprompt import BootstrapFewShot
import litellm
import os

LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")

BLOOM_GUIDANCE_PROMPTS = {
    "Remembering": "This is a Remembering-level question. Give a brief explanation that helps the student recall key facts or terms to answer the question. Include a mnemonic or simple strategy to remember.",
    "Understanding": "This is an Understanding-level question. Explain how the student can paraphrase or explain the concept in their own words, and walk through the reasoning that leads to the answer.",
    "Applying": "This is an Applying-level question. Describe how the student should use prior knowledge in a new situation. Walk through the steps of applying a concept to solve the problem.",
    "Analyzing": "This is an Analyzing-level question. Guide the student to break down the concept into parts, find patterns, or relationships. Walk through how to think critically to reach the answer.",
    "Evaluating": "This is an Evaluating-level question. Help the student assess or justify a position or solution. Walk through the logic used to check or critique the answer.",
    "Creating": "This is a Creating-level question. Guide the student through combining ideas to produce something new. Walk through the steps of designing or generating a solution."
}

def get_training_data():
    return [
        # REMEMBERING
        dspy.Example(
            tutor_prompt=build_explanation_prompt(
                "What is the chemical symbol for sodium?",
                "Na",
                "Remembering"
            ),
            explanation="This is a Remembering-level question. To answer it, you need to recall the symbol for sodium from the periodic table. A helpful strategy is to remember that some elements use Latin roots. Sodium’s Latin name is 'Natrium', which is why its symbol is 'Na'. A quick mnemonic: 'Na – Not Apparent from English!' to help remember it doesn't match the English spelling."
        ).with_inputs("question", "answer", "bloom_level", "tutor_prompt"),

        # UNDERSTANDING
        dspy.Example(
            tutor_prompt=build_explanation_prompt(
                "Explain why the heart has four chambers.",
                "The heart has four chambers to separate oxygen-rich and oxygen-poor blood, ensuring efficient circulation throughout the body.",
                "Understanding"
            ),
            explanation="This is an Understanding-level question. To answer it, try to explain in your own words what each chamber does. The right side of the heart handles blood returning from the body and sends it to the lungs to pick up oxygen. The left side receives the oxygenated blood and pumps it to the body. Walking through the process helps you understand how the heart keeps the oxygen-rich and oxygen-poor blood separate to work efficiently."
        ).with_inputs("question", "answer", "bloom_level", "tutor_prompt"),

        # APPLYING
        dspy.Example(
            tutor_prompt=build_explanation_prompt(
                "Use the Pythagorean theorem to find the length of the hypotenuse in a right triangle with legs of 3 and 4 units.",
                "The hypotenuse is 5 units. Using a² + b² = c²: 3² + 4² = 9 + 16 = 25, so √25 = 5.",
                "Applying"
            ),
            explanation="This is an Applying-level question. Start by recognizing that this is a right triangle, so the Pythagorean theorem applies: a² + b² = c². Use the values: a = 3 and b = 4. Compute 3² = 9 and 4² = 16, then add them: 9 + 16 = 25. Now take the square root of 25 to find the hypotenuse: √25 = 5. This shows how to use prior knowledge in a new situation."
        ).with_inputs("question", "answer", "bloom_level", "tutor_prompt"),

        # ANALYZING
        dspy.Example(
            tutor_prompt=build_explanation_prompt(
                "Compare mitosis and meiosis in terms of genetic variation and number of resulting cells.",
                "Mitosis creates two identical cells; meiosis creates four genetically unique cells. Mitosis is for growth, meiosis for reproduction.",
                "Analyzing"
            ),
            explanation="This is an Analyzing-level question. Start by breaking down each process: mitosis results in two identical cells, and is used for growth and repair. Meiosis results in four cells with half the DNA, all genetically different, and is used for reproduction. Notice the patterns: mitosis keeps the DNA the same; meiosis increases variation. You're analyzing by comparing structure, outcome, and function to understand deeper relationships."
        ).with_inputs("question", "answer", "bloom_level", "tutor_prompt"),

        # EVALUATING
        dspy.Example(
            tutor_prompt=build_explanation_prompt(
                "Which programming language would you recommend for beginners and why?",
                "Python, because its simple syntax makes it easy to learn, and it's widely used in various fields from web development to AI.",
                "Evaluating"
            ),
            explanation="This is an Evaluating-level question. To answer, you must assess programming languages based on clarity, ease of learning, and real-world use. Python stands out due to its readable syntax and broad application. For example, 'print(\"Hello\")' in Python is much simpler than in Java or C++. You're making a justified recommendation by comparing choices and applying logical criteria like simplicity and flexibility."
        ).with_inputs("question", "answer", "bloom_level", "tutor_prompt"),

        # CREATING
        dspy.Example(
            tutor_prompt=build_explanation_prompt(
                "Design an experiment to test the effect of light on plant growth.",
                "Place identical plants under different light conditions (sunlight, LED, darkness) and measure growth over 2 weeks to compare results.",
                "Creating"
            ),
            explanation="This is a Creating-level question. Start by defining your goal — testing how light affects growth. Then combine your knowledge of experimental design: choose a variable (light type), control others (same plant species, water, soil), and decide on a measurable outcome (growth in cm). By generating this plan, you're combining ideas to build a new solution from scratch — the hallmark of creative thinking."
        ).with_inputs("question", "answer", "bloom_level", "tutor_prompt")
    ]


#signatures
class ExplanationSignature(dspy.Signature):
    tutor_prompt = dspy.InputField()
    explanation = dspy.OutputField(desc="A tutor-style explanation with strategy and walkthrough")

#modules
class ExplanationGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(ExplanationSignature)

    def forward(self, tutor_prompt):
        return self.generate(tutor_prompt=tutor_prompt)

def build_explanation_prompt(question, answer, bloom_level):
    strategy = BLOOM_GUIDANCE_PROMPTS.get(bloom_level, "")
    return (
        f"{strategy}\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}"
    )

def compile_model():
    examples = get_training_data()  
    booster = BootstrapFewShot(metric=lambda e, p, _: 1, max_bootstrapped_demos=6)
    return booster.compile(ExplanationGenerator(), trainset=examples)


def main():
    parser = argparse.ArgumentParser(description="Generate Bloom-level tutor-style explanations via prompt injection")
    parser.add_argument("--input", type=str, required=True)
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

    ####
    with open(args.input, "r") as f:
        raw_data = json.load(f)

    

    
    model = compile_model()
    count=0

    for item in raw_data:
        question = item["question"]
        answer = item["answer"]
        bloom_level = item["bloom_level"]

        prompt = build_explanation_prompt(question, answer, bloom_level)
        result = model(tutor_prompt=prompt)
        explanation = result.explanation.strip()

        item["explanation"] = explanation

        count += 1

    
    with open(args.output, "w") as f:
        json.dump(raw_data, f, indent=2)


    print(f"\n Generated explanations for {count} entries.")
    print(f" Output saved to: {args.output}")

if __name__ == "__main__":
    main()

