import json
import os
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import GPTModel
from deepeval import evaluate
import openai
from openai import OpenAI
from deepeval.models.base_model import DeepEvalBaseLLM
import requests
import litellm, asyncio
import argparse

LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")

parser = argparse.ArgumentParser(description="Evaluate explanations using G-Eval metrics.")

parser.add_argument(
    "-i", "--input", 
    type=str, 
    required=True, 
    help="Path to the input JSON file containing explanations."
)

parser.add_argument(
    "-o", "--output", 
    type=str, 
    required=True, 
    help="Path to save the output evaluation JSON file."
)

args = parser.parse_args()


# Load data
with open(args.input, "r") as f:
    data = json.load(f)


class LiteLLMWrapper(DeepEvalBaseLLM):
    def __init__(self, model_name: str, api_key: str, api_base: str = None):
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.model = self.load_model()

    def get_model_name(self) -> str:
        return self.model_name

    def load_model(self):
        litellm.api_key = self.api_key
        if self.api_base:
            litellm.api_base = self.api_base
        return litellm

    def generate(self, prompt: str) -> str:
        response = self.model.completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content']

    async def a_generate(self, prompt: str) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt)

custom_llm = LiteLLMWrapper(
    model_name="openai/claude-3.7-sonnet",  
    api_key= LITELLM_API_KEY,
    api_base= LITELLM_BASE_URL 
)

def make_input(item):
    return (
        f"Question: {item['question']}\n"
        f"Answer: {item['answer']}\n"
        f"Bloom Level: {item['bloom_level']}\n"
    )

# Metrics 
correctness_metric = GEval(
    name="Correctness",
    
    criteria="Evaluate whether the explanation is factually accurate and logically consistent with the correct answer. The explanation must not contain any incorrect or misleading information. It should support or justify the correct answer, either directly or indirectly. Elaboration is acceptable as long as it aligns with the correct answer and does not introduce confusion or contradictions.  It is acceptable if the correct answer is clearly implied, even if it is not explicitly stated; do not penalize for lack of explicit restatement.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=custom_llm,
    verbose_mode= True
)


bloom_alignment_metric = GEval(
    name="Bloom Alignment",

    criteria="Assess whether the explanation demonstrates the thinking style or cognitive demand associated with the specified Bloom’s level (e.g., factual recall for Remembering, conceptual explanation for Understanding, real-world application for Applying). Do not evaluate for factual correctness or instructional quality.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=custom_llm,
    verbose_mode=True
)

pedagogy_metric = GEval(
    name="Pedagogical Soundness",
    
    criteria="""
    Evaluate how well the explanation functions as a teaching tool. Consider its clarity, organization, engagement. Place emphasis on how effectively it guides the student's thought process towards arriving at the answer or verifying it rather than just providing details. Do not evaluate for factual accuracy or alignment with the specified Bloom level.
    Some examples of explanations which effectively work as teaching tools are:
    -Question: 12*6?
     Answer: 72
     Explanation: To answer this question, you need to multiply 12 with 6. In order to make this easier, you can do 10*6 and then add 6+6. The result is 72.
     Evaluation:This explanation is clear, organized and easy to understand by someone who is in elementary school. It effectively guides the learner towards the answer by explaining the process and giving tricks to simplify it.
    -Question:Explain the phenomenon of evaporation.
     Answer:Evaporation is when liquid water turns into gas (water vapor) and mixes into the air. It happens because heat gives energy to surface molecules, allowing them to escape into the air as gas.
     Explanation: Imagine you spill a little bit of water on a table. After a while, it’s gone — not because someone wiped it, but because it evaporated. That means the liquid water changed into gas (called water vapor) and mixed into the air. This happens because heat gives energy to the water molecules, and some at the surface get enough energy to break away and become gas.
     Evaluation: This explanation is clear and organized, provides a hands-on example to make the concept clear and keeps an engaging tone.
    -Question: Create a math word problem involving percentages.
     Answer:Sarah bought a jacket that was originally priced at $80. It was on sale for 25% off. After the discount, she also had to pay 8% sales tax on the discounted price. How much did Sarah pay in total for the jacket?
     Explanation: To create a math word problem involving percentages, start with a character—like Sarah—to keep it relatable. Choose a real-life scenario, such as shopping. For example, say Sarah finds a jacket that costs $80. It's on sale for 25% off, and she has to pay 8% sales tax on the discounted price. Use these numbers to build the problem: Sarah found a jacket originally priced at $80. It was 25% off during a sale. After the discount, she had to pay 8% sales tax. How much did Sarah pay in total?
     Evaluation: This explanation effectively guides the student towards creating a problem. It is concise and clear and keeps an engaging tone.
    -Question: Two companies are creating fitness apps: App A stores all user data locally on the device, while App B uses cloud storage and machine learning to offer personalized advice. Which approach provides a better balance between user experience and data security? Evaluate both options and justify your answer with pros and cons.
     Answer: App A offers better data security since information stays on the user’s device, reducing the risk of breaches. However, App B provides a more advanced user experience through personalized insights using machine learning. Overall, App B may be more effective for user engagement, but App A is better for privacy. The better choice depends on whether the user values security over smart features
     Explanation: Start by identifying the main difference: App A stores data locally (safer, but limited features), while App B uses the cloud and machine learning (more features, but possible privacy risks). Think about what users care about—do they want advanced, personalized help (like workout tips) or do they care more about keeping their data private? List the pros and cons of each: App A is more secure, but less smart; App B is smarter, but may share or store data online. Then, decide which matters more in the situation and explain why. There’s no one “right” answer—what matters is your reasoning.   
    Evaluation: This explanation effectively guides the student on how evaluate the given choices and walks through the answer in a clear and structured way while keeping an engaging tone.
    """,
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=custom_llm,
    verbose_mode=True
)
# Evaluation loop 
results = []
for item in data:
    test_case = LLMTestCase(
        input=make_input(item),
        actual_output=item["explanation"], #COMMENT OUT FOR PLANNING
        #actual_output=item["instruction_plan"], UNCOMMENT FOR PLANNING
    )

    
    correctness_metric.measure(test_case)
    bloom_alignment_metric.measure(test_case)
    pedagogy_metric.measure(test_case)

    
    result = {
        "question": item["question"],
        "answer": item["answer"],
        "explanation": item["explanation"],
        #"instruction plan":item["instruction_plan"], UNCOMMENT FOR PLANNING
        "topic": item["topic"],
        "difficulty": item["difficulty"],
        "bloom_level": item["bloom_level"],
        #"predicted bloom level":item["predicted_bloom_level"], UNCOMMENT FOR AQ
        #"bloom reasoning":item["bloom_reasoning"], UNCOMMENT FOR AQ
        "evaluation": {
            "correctness": {
                "score": correctness_metric.score,
                "reason": correctness_metric.reason
            },
            "bloom_alignment": {
                "score": bloom_alignment_metric.score,
                "reason": bloom_alignment_metric.reason
            },
            "pedagogical_soundness": {
                "score": pedagogy_metric.score,
                "reason": pedagogy_metric.reason
            },
            "overall_score": round((
                correctness_metric.score +
                bloom_alignment_metric.score +
                pedagogy_metric.score
            ) / 3, 2)
        }
    }
    results.append(result)

avg = lambda key: round(sum(r["evaluation"][key]["score"] for r in results) / len(results), 2)
summary = {
    "average_correctness": avg("correctness"),
    "average_bloom_alignment": avg("bloom_alignment"),
    "average_pedagogical_soundness": avg("pedagogical_soundness"),
    "average_overall_score": round(sum(r["evaluation"]["overall_score"] for r in results) / len(results), 2)
}

output = {
    "summary": summary,
    "evaluations": results
}

with open(args.output, "w") as f:
    json.dump(output, f, indent=2)


print("G-Eval evaluation completed and saved.")
