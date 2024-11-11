# TODO: リファクタリング・ruffの有効化
# issue: https://github.com/lovelovetrb/xlstm-lm/issues/6
# ruff: noqa
import argparse

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from dacite import Config as DaciteConfig
from dacite import from_dict
from omegaconf import OmegaConf
from transformers import AutoTokenizer, set_seed

from src.experiment.setup.model import setup_model
from src.cfg.load_yaml_cfg import load_config
from src.utils import dist_cleanup, dist_setup, get_logger


class TextGenerator:
    def __init__(self, model, tokenizer, eos_token_id):
        self.model = model
        self.eos_token_id = eos_token_id

    def generate_text(
        self,
        start_sequence,
        max_length=100,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        do_sample=True,
        num_return_sequences=1,
        use_beam_search=False,
        num_beams=5,
        length_penalty=1.0,
    ):
        batch_size = start_sequence.shape[0]
        vocab_size = self.model.config.vocab_size

        if use_beam_search:
            beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=start_sequence.device)
            beam_scores[:, 1:] = -1e9
            beam_sequences = start_sequence.repeat(1, num_beams).view(batch_size * num_beams, -1)
            done = [False for _ in range(batch_size)]

            for _ in range(max_length):
                model_inputs = beam_sequences

                if model_inputs.shape[1] > max_length:
                    model_inputs = model_inputs[:, -max_length:]
                    print(f"Truncated input to length: {model_inputs.shape[1]}")

                outputs = self.model(model_inputs)
                next_token_logits = outputs[:, -1, :]

                del model_inputs

                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size * num_beams):
                        for previous_token in set(beam_sequences[i].tolist()):
                            next_token_logits[i, previous_token] /= repetition_penalty

                # Calculate log probabilities
                next_token_scores = F.log_softmax(next_token_logits, dim=-1)

                # Add the log probabilities to the beam scores
                next_token_scores = next_token_scores + beam_scores.view(-1, 1)

                # Reshape scores into (batch_size, num_beams * vocab_size)
                next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

                # Get the top-k scores and corresponding token indices
                topk_scores, topk_indices = torch.topk(next_token_scores, num_beams, dim=1)

                # Convert indices to token IDs and beam indices
                topk_token_ids = topk_indices % vocab_size
                beam_indices = topk_indices // vocab_size

                # Update sequences, scores, and check for completed sequences
                new_sequences = []
                new_scores = []
                for batch_idx in range(batch_size):
                    if done[batch_idx]:
                        new_sequences.extend([beam_sequences[batch_idx * num_beams + i] for i in range(num_beams)])
                        new_scores.extend([beam_scores[batch_idx][i] for i in range(num_beams)])
                        continue

                    for beam_idx in range(num_beams):
                        token_id = topk_token_ids[batch_idx][beam_idx]
                        score = topk_scores[batch_idx][beam_idx]
                        beam_idx = beam_indices[batch_idx][beam_idx]

                        seq = torch.cat(
                            [
                                beam_sequences[batch_idx * num_beams + beam_idx],
                                token_id.unsqueeze(0),
                            ],
                            dim=-1,
                        )
                        new_sequences.append(seq)
                        new_scores.append(score)

                        if token_id.item() == self.eos_token_id:
                            print("eos_token_id is generated.")
                            done[batch_idx] = True

                beam_sequences = torch.stack(new_sequences).view(batch_size * num_beams, -1)
                beam_scores = torch.tensor(new_scores, device=start_sequence.device).view(batch_size, num_beams)

                if all(done):
                    break

            # Select the best beam for each batch
            generated_sequences = []
            for batch_idx in range(batch_size):
                best_score, best_idx = beam_scores[batch_idx].max(dim=0)
                generated_sequences.append(beam_sequences[batch_idx * num_beams + best_idx].cpu().tolist())

        else:
            # Original sampling-based generation code (as in the previous version)
            current_sequences = start_sequence.repeat(num_return_sequences, 1)
            generated_sequences = []

            for _ in range(max_length):
                with torch.no_grad():
                    outputs = self.model(current_sequences)

                next_token_logits = outputs[:, -1, :] / temperature

                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(num_return_sequences):
                        for previous_token in set(current_sequences[i].tolist()):
                            next_token_logits[i, previous_token] /= repetition_penalty

                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float("-inf")

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float("-inf")

                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

                current_sequences = torch.cat([current_sequences, next_tokens], dim=-1)

                # Check if any sequence has generated the EOS token
                eos_generated = (next_tokens.squeeze(-1) == self.eos_token_id).any(dim=0)

                if eos_generated:
                    for idx, seq in enumerate(current_sequences):
                        if seq[-1] == self.eos_token_id or _ == max_length - 1:
                            generated_sequences.append(seq.tolist())
                            current_sequences = current_sequences[torch.arange(current_sequences.size(0)) != idx]
                            if len(current_sequences) == 0:
                                return generated_sequences

            # If we've reached here, it means we've hit max_length for all sequences
            for seq in current_sequences:
                generated_sequences.append(seq.tolist())

        return generated_sequences


def main(rank, world_size, config):
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.basic.device_ids))
    logger = get_logger(__name__)
    set_seed(config.basic.seed)
    dist_setup(rank, world_size, logger)
    torch.cuda.set_device(rank)

    model = setup_model(config, rank, config.basic.model_weight_path)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.name)
    tokenizer.add_special_tokens({"pad_token": "[PAD]", "bos_token": "[BOS]", "eos_token": "[EOS]"})
    eos_token_id = tokenizer.eos_token_id

    text = "[BOS]静岡市には"
    tokenized_text = tokenizer(text, return_tensors="pt")

    generated_text = Generator(model, eos_token_id).generate_text(
        model,
        tokenized_text["input_ids"].to(config.basic.device),
        eos_token_id,
        max_length=config.dataset.max_seq_length,
        temperature=0.6,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=1,
        use_beam_search=True,
        num_beams=5,
        length_penalty=1.5,
    )
    if rank == 0:
        print(tokenizer.decode(generated_text[0]))

    dist_cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    config = load_config(parser.parse_args().config_path)
    WORLD_SIZE = torch.cuda.device_count()

    mp.spawn(main, args=(WORLD_SIZE, config), nprocs=WORLD_SIZE, join=True)
