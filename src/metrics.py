import torch

def compute_logit_diff(logits: torch.Tensor, correct_id: int, incorrect_id: int) -> float:
    """Computes difference in final logits between correct and incorrect token IDs."""
    final_logits = logits[0, -1, :]
    return (final_logits[correct_id] - final_logits[incorrect_id]).item()


def evaluate_prompts(model, data: list) -> list:
    """Runs dataset prompts through model, checking correctness and logit differences."""
    results = []
    for item in data:
        logits, _ = model.run_with_cache(item["prompt"])
        last_logit = logits[0, -1, :]
        
        pred_token_id = torch.argmax(last_logit).item()
        pred_str = model.to_string(pred_token_id)
        
        target_id = model.to_single_token(item["target"])
        corrupt_id = model.to_single_token(item["corrupt"])
        
        logit_diff = (last_logit[target_id] - last_logit[corrupt_id]).item()
        is_correct = (pred_str.strip() == item["target"].strip())
        
        results.append({
            "prompt": item["prompt"],
            "difficulty": item.get("difficulty", "N/A"),
            "logit_diff": logit_diff,
            "correct": is_correct
        })
    return results
