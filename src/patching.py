import torch

def compute_logit_diff(logits, correct_id, incorrect_id):
    """Computes the difference in logits between correct and incorrect tokens at final sequence position."""
    final_logits = logits[0, -1, :]
    return (final_logits[correct_id] - final_logits[incorrect_id]).item()

def evaluate_prompts(model, data):
    """Evaluates arithmetic prompt dataset and calculates logit diffs and correctness."""
    results = []
    for item in data:
        logits, cache = model.run_with_cache(item["prompt"])
        last_logit = logits[0, -1, :]
        pred_token_id = torch.argmax(last_logit).item()
        pred_str = model.to_string(pred_token_id)
        target_id = model.to_single_token(item["target"])
        corrupt_id = model.to_single_token(item["corrupt"])
        logit_diff = (last_logit[target_id] - last_logit[corrupt_id]).item()
        is_correct = (pred_str.strip() == item["target"].strip())
        results.append({
            "prompt": item["prompt"],
            "difficulty": item["difficulty"],
            "logit_diff": logit_diff,
            "correct": is_correct
        })
    return results
