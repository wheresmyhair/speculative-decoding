from typing import Tuple, List, Union, Dict
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm


def predict_next_n_tokens(model: T5ForConditionalGeneration, input_ids: torch.Tensor, num_new_tokens: int = 1):
    """Call model.generate() and return the output
    """
    outputs = model.generate(input_ids,
                             max_new_tokens = num_new_tokens, 
                             return_dict_in_generate = True, 
                             output_scores = True)
    return outputs


def score_to_prob(scores: torch.Tensor, temperature: float = 1., top_p: float = 1., use_argmax: bool = False) -> torch.Tensor:
    """Convert scores (NOT softmaxed tensor) to probabilities with support for temperature, top-p sampling, and argmax.

    Parameters
    ----------
    scores : torch.Tensor
        Input scores.
    temperature : float, optional
        Temperature parameter for controlling randomness. Higher values make the distribution more uniform, lower values make it peakier, by default 1.0
    top_p : float, optional
        Top-p sampling parameter for controlling the cumulative probability threshold, by default 1.0 (no threshold)
    use_argmax : bool, optional
        Whether to use argmax instead of temperature or top-p sampling, by default False

    Returns
    -------
    torch.Tensor
        Probability distribution after adjustments.
    """
    assert temperature > 0.0
    assert 0.0 < top_p <= 1.0
    if use_argmax:
        final_prob = F.one_hot(scores.argmax(dim=1), num_classes=scores.size(1)).float()
    else:
        if temperature != 1.0:
            scores /= temperature
        if top_p < 1.0:
            sorted_scores, _ = torch.sort(scores, descending=True)
            cumulative_probs = torch.cumsum(sorted_scores.softmax(dim=1), dim=1)
            mask = cumulative_probs <= top_p
            if mask.any():
                thresholded_probs = cumulative_probs * mask
                thresholded_probs = thresholded_probs / thresholded_probs.sum(dim=1, keepdim=True)
                final_prob = torch.zeros_like(scores)
                final_prob.scatter_add_(1, sorted_scores.argsort(dim=1), thresholded_probs)
            else:
                final_prob = scores.softmax(dim=1)
        else:
            final_prob = scores.softmax(dim=1)

    return final_prob


def sample(prob: torch.Tensor, num_samples: int = 1) -> Dict:
    """Sample from a tensor of probabilities
    """
    sampled_indices = torch.multinomial(prob, num_samples=num_samples) 
    return {'token': sampled_indices, 'prob': prob.gather(dim=1, index=sampled_indices), 'all_prob': prob}


def autoregressive_sampling(model, input_ids: torch.Tensor, num_new_tokens: int = 5) -> Dict:
    """Ref: [arXiv:2211.17192v2](https://arxiv.org/abs/2211.17192) Section 2.2
    """
    sequences = input_ids
    new_tokens = []
    
    for _ in range(num_new_tokens):
        pred = predict_next_n_tokens(model=model, input_ids=sequences, num_new_tokens=1) # predict next one token
        prob = score_to_prob(pred.scores[0])
        sampled = sample(prob=prob, num_samples=1)
        new_tokens.append(sampled)
        sequences = torch.cat([sequences, sampled['token']], dim=1)
        
    return {"sequences": sequences, "new_tokens": new_tokens}


def speculative_sampling(prefix_ids: torch.Tensor, 
                        draft_model: T5ForConditionalGeneration, 
                        target_model_list: List[T5ForConditionalGeneration]) -> torch.Tensor:
    """Ref: [arXiv:2211.17192v2](https://arxiv.org/abs/2211.17192)
    """
    gamma = len(target_model_list) - 1 # One of the models is left for the best case.
    assert gamma > 0
    # STEP 1: Sample γ guesses x1,...,γ from Mq (draft model) autoregressively
    len_prefix_ids = prefix_ids.shape[1]
    outputs_draft = autoregressive_sampling(model=draft_model, input_ids=prefix_ids, num_new_tokens=gamma)


    # STEP 2: Run Mp (target model) in parallel
    # generate sequences [prefix, prefix+x1, prefix+x1+x2, ..., prefix+x1+x2+...+xγ]
    all_sequences = [outputs_draft['sequences'][:,:len_prefix_ids+x] for x in range(0, gamma+1)] 
    with ThreadPoolExecutor() as executor:
        predictions = executor.map(predict_next_n_tokens, target_model_list, all_sequences, [1]*(gamma+1))
        # TODO: rearrange such that once reject a sample, stop running the rest models.
    results = list(predictions)


    # STEP 3: Determine the number of accepted guesses n
    accepted = [False] * gamma
    for i in range(gamma):
        draft_token_id = outputs_draft['new_tokens'][i]['token']
        draft_token_prob = outputs_draft['new_tokens'][i]['prob']
        target_token_prob = score_to_prob(results[i].scores[0])[0, draft_token_id]

        # reject the sample with probability 1 - p(x)/q(x)
        if torch.rand_like(target_token_prob) > target_token_prob/draft_token_prob:
            break
        
        accepted[i] = True
        
    print(f"Accepted: {sum(accepted)}/{gamma}")


    # STEP 4: Adjust the distribution from Mp if needed
    if not all(accepted):
        target_all_prob = score_to_prob(results[i].scores[0])
        draft_all_prob = outputs_draft['new_tokens'][i]['all_prob']
        adjusted_prob = torch.max(torch.zeros_like(target_all_prob), target_all_prob - draft_all_prob)
        prob = adjusted_prob / adjusted_prob.sum(dim=1, keepdim=True)
    else:
        prob = score_to_prob(results[-1].scores[0])


    # STEP 5: Return one token from Mp, and n tokens from Mq
    token_from_target_model = sample(prob)['token']
    flnal_sequences = torch.concat([all_sequences[sum(accepted)], token_from_target_model], dim=1)


    return flnal_sequences

