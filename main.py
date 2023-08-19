import torch
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

from spec_decoding import speculative_sampling


gamma = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer_draft = T5Tokenizer.from_pretrained("google/flan-t5-small")
tokenizer_target = T5Tokenizer.from_pretrained("google/flan-t5-base")
model_draft = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small", device_map='auto')
models_target = [T5ForConditionalGeneration.from_pretrained("google/flan-t5-base", device_map='auto') for _ in tqdm(range(gamma+1))] # In the best case, i.e., all guesses are accepted, we need to run the target model gamma+1 times


if __name__ == "__main__":
    prefix = "translate English to German: How old"
    prefix_ids = tokenizer_draft.encode(prefix, return_tensors="pt").to(device)
    max_new_tokens=50
    num_generated_new_tokens = 0
    while num_generated_new_tokens < max_new_tokens:
        res = speculative_sampling(prefix_ids, model_draft, models_target)
        print(tokenizer_draft.decode(res[0], skip_special_tokens=False))
        print(res)
        num_generated_new_tokens += len(res[0]) - len(prefix_ids[0])
        prefix_ids = res

