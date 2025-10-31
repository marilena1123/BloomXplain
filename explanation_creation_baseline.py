import argparse
import json
import dspy
from dspy.teleprompt import BootstrapFewShot
import litellm
import os

LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")

def build_explanation_prompt(question, answer):
    return (
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        f"Create an explanation for why this is the correct answer."
    )
def get_training_data():
    return [
        dspy.Example(
            tutor_prompt=build_explanation_prompt(
                "What is the chemical symbol for sodium?",
                "Na"
            ),
            explanation="The chemical symbol for sodium is 'Na'. This comes from its Latin name, 'Natrium'."
        ).with_inputs("question", "answer", "tutor_prompt"),

        dspy.Example(
            tutor_prompt=build_explanation_prompt(
                "Explain why the heart has four chambers.",
                "The heart has four chambers to separate oxygen-rich and oxygen-poor blood, ensuring efficient circulation throughout the body."
            ),
            explanation="The heart has two sides with two chambers each. One side sends blood to the lungs to get oxygen, and the other side pumps oxygen-rich blood to the rest of the body. This setup keeps the two types of blood from mixing."
        ).with_inputs("question", "answer", "tutor_prompt"),

        dspy.Example(
            tutor_prompt=build_explanation_prompt(
                "Use the Pythagorean theorem to find the length of the hypotenuse in a right triangle with legs of 3 and 4 units.",
                "The hypotenuse is 5 units. Using a² + b² = c²: 3² + 4² = 9 + 16 = 25, so √25 = 5."
            ),
            explanation="To find the hypotenuse, use the formula a² + b² = c². Plug in the values: 3² = 9 and 4² = 16. Add them to get 25, and take the square root. The result is 5."
        ).with_inputs("question", "answer", "tutor_prompt"),

        dspy.Example(
            tutor_prompt=build_explanation_prompt(
                "Compare mitosis and meiosis in terms of genetic variation and number of resulting cells.",
                "Mitosis creates two identical cells; meiosis creates four genetically unique cells. Mitosis is for growth, meiosis for reproduction."
            ),
            explanation="Mitosis creates two cells that are exactly the same, useful for things like repairing tissue. Meiosis makes four cells that are all different, which helps with genetic diversity in reproduction."
        ).with_inputs("question", "answer", "tutor_prompt"),

        dspy.Example(
            tutor_prompt=build_explanation_prompt(
                "Which programming language would you recommend for beginners and why?",
                "Python, because its simple syntax makes it easy to learn, and it's widely used in various fields from web development to AI."
            ),
            explanation="Python is recommended because it’s easy to read and write. It’s also used in many areas like building websites, automating tasks, and working with data."
        ).with_inputs("question", "answer", "tutor_prompt"),

        dspy.Example(
            tutor_prompt=build_explanation_prompt(
                "Design an experiment to test the effect of light on plant growth.",
                "Place identical plants under different light conditions (sunlight, LED, darkness) and measure growth over 2 weeks to compare results."
            ),
            explanation="To test this, you can place the same type of plant in different lighting conditions (sunlight, LED, darkness). Over a few weeks, measure how much each plant grows to see if light affects growth."
        ).with_inputs("question", "answer", "tutor_prompt")
    ]

# Signatures
class ExplanationSignature(dspy.Signature):
    tutor_prompt = dspy.InputField()
    explanation = dspy.OutputField(desc="A tutor-style explanation")

# Module
class ExplanationGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(ExplanationSignature)

    def forward(self, tutor_prompt):
        return self.generate(tutor_prompt=tutor_prompt)


def compile_model():
    examples = get_training_data()
    booster = BootstrapFewShot(metric=lambda e, p, _: 1, max_bootstrapped_demos=6)
    return booster.compile(ExplanationGenerator(), trainset=examples)


def main():
    parser = argparse.ArgumentParser(description="Generate tutor-style explanations without Bloom guidance.")
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

    with open(args.input, "r") as f:
        raw_data = json.load(f)

    model = compile_model()

    count = 0

    for item in raw_data:
        question = item["question"]
        answer = item["answer"]

        prompt = build_explanation_prompt(question, answer)
        result = model(tutor_prompt=prompt)
        explanation = result.explanation.strip()

        item["explanation"] = explanation

        count += 1

    with open(args.output, "w") as f:
        json.dump(raw_data, f, indent=2)

    print(f"\nGenerated explanations for {count} entries.")
    print(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()

