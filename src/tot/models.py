import sys
import time
import torch
import re
from prompts.text import *

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline



class LLM: 
    def __init__(self, model_name='Meta-Llama-3-8B'):
        # Set up the tokenizer and pipeline
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./kaggle/working/")

        print('** LLM init, checking tokenizer:', self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    # Temperature: Higher values will make the output more random, while lower values will make it more focused and deterministic.
    # Top_p: An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    # top_k: Introduces random sampling for generated tokens by randomly selecting the next token from the k most likely options.
    def llama(self, user_prompt, system_prompt=None, max_tokens=512, do_sample=True, beams=3, n=1, top_k=50, top_p=0.5, temperature=1.0):
        if system_prompt is None:
            system_prompt = "You are a chatbot who always responds to the question!"
        # else:
        #    print('**llama Using system_prompt:', system_prompt)

        # Concatenate system and user prompts directly
        combined_prompt = f"{system_prompt}\n{user_prompt}"


        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": combined_prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to("cuda")

        generated_ids = self.model.generate(
            **model_inputs,
            temperature=temperature,
            max_new_tokens=max_tokens,
            num_beams=beams
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        #response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        decoded_responses = []
        for response in generated_ids:
            #response = output[input_ids.shape[-1]:]
            decoded_response = self.tokenizer.decode(response, skip_special_tokens=True)
            decoded_responses.append(decoded_response)

        print('** llama: combined_prompt', combined_prompt)
        print('** llama: decoded_responses', decoded_responses)

        return decoded_responses


    def llama_usage(self, backend='7B'):
        return {'completion_tokens': 0, 'prompt_tokens': 0, 'cost': 0}

