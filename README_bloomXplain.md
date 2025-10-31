-Dataset:  

The "Dataset" folder contains our benchmark dataset. Each file is named using the format:
bloom_qa_{difficulty}_{topic}.json (single Bloom-level questions).  

bloom_qa_multi_{topic}.json (multi Bloom-level questions).


-Generating Results (Table 2):  

To generate explanations using the BAQ method, run the script explanation_creation_baq.py with the following arguments: input and output json files (the LLM is hardcoded in the script, to change it, find and modify the corresponding line)  

To generate explanations using the AQ method, run the script explanation_creation_aq.py with the following arguments: input and output json files (the LLM is hardcoded in the script, to change it, find and modify the corresponding line)  

To generate explanations using the BASELINE method, run the script explanation_creation_baseline.py with the following arguments: input and output json files (the LLM is hardcoded in the script, to change it, find and modify the corresponding line)  

To evaluate, run geval.py with the following arguments: input and output json files. The script evaluates BAQ-style methods by default. For other methods, comment/uncomment the relevant lines in geval.py based on the comments.  


-Generating Results (Table 4):  

To generate BAQ explanations, modify explanation_creation_baq.py to use the corresponding prompts and comment out the line where the answer is provided to the model.  

To generate CoT explanations, run CoT.py with the following arguments: input_file, output_file, task ((the LLM is hardcoded in the script, to change it, find and modify the corresponding line)
).  


-Generating results (Table 5)  

To generate plans, modify explanation_creation_baq.py accordingly: use the corresponding prompts and few-shot examples and comment out the where the answer is provided to the model.  

To evaluate, run geval.py using the corresponding criteria for the metrics.


-The above steps refer to the results included in the main sections of the paper. To maintain simplicity and readability, configurations for other ablations are not directly included but can be easily set up with minor adjustments.  


-We are using LITELLM proxy. Before running the scripts, please set the following environment variables: LITELLM_API_KEY and LITELLM_BASE_URL.


-The "few shot examples" folder contains JSON files with the few-shot examples used in our methods for review and reference.  

