# TODO: リファクタリング・ruffの有効化
# issue: https://github.com/lovelovetrb/xlstm-lm/issues/6
# ruff: noqa
import argparse

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from transformers import set_seed

from src.experiment.setup.model import setup_model
from src.experiment.setup.tokenizer import setup_tokenizer
from src.cfg.load_yaml_cfg import load_config
from src.utils import dist_cleanup, dist_setup, get_logger


class TextGenerator:
    def __init__(self, model, eos_token_id, rank, tokenizer):
        self.model = model
        self.eos_token_id = eos_token_id
        self.rank = rank
        self.tokenizer = tokenizer
        self.logger = get_logger("generator")

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
            self.logger.info("ビームサーチあり")
            beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=self.rank)
            beam_scores[:, 1:] = -1e9
            beam_sequences = start_sequence.repeat(1, num_beams).view(batch_size * num_beams, -1)
            done = [False for _ in range(batch_size)]

            for _ in range(max_length):
                model_inputs = beam_sequences

                if model_inputs.shape[1] > max_length:
                    model_inputs = model_inputs[:, -max_length:]

                with torch.no_grad():
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
                beam_scores = torch.tensor(new_scores, device=self.rank).view(batch_size, num_beams)

                if all(done):
                    break

            # Select the best beam for each batch
            generated_sequences = []
            for batch_idx in range(batch_size):
                best_score, best_idx = beam_scores[batch_idx].max(dim=0)
                generated_sequences.append(beam_sequences[batch_idx * num_beams + best_idx].cpu().tolist())

            return generated_sequences

        else:
            self.logger.info("ビームサーチなし")
            with torch.no_grad():
                current_sequence = start_sequence
                for _ in range(max_length - 1):
                    # モデルの出力を取得

                    # print(self.tokenizer.decode(current_sequence[0]))
                    outputs = self.model(current_sequence)
                    next_token_logits = outputs[:, -1, :] / temperature
                    # 次のトークンを確率的にサンプリング
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    # 現在のシーケンスに追加
                    current_sequence = torch.cat([current_sequence, next_token], dim=-1)

                    # EOS トークンが生成されたら終了
                    if next_token.item() == self.eos_token_id:
                        break

            return current_sequence.tolist()


def main(rank, world_size, config, start_sentence):
    logger = get_logger(__name__)
    set_seed(config.basic.seed)
    dist_setup(rank, world_size, logger)

    model = setup_model(config, rank, config.basic.model_weight_path)

    tokenizer = setup_tokenizer(config.tokenizer.name)
    eos_token_id = tokenizer.eos_token_id

    # start_sentence = "電子機器で使用される最も主要な電子回路基板の事をなんと言う？ 選択肢0: 掲示板 選択肢1: パソコン 選択肢2: マザーボード, 選択肢3: ハードディスク, 選択肢4: まな板 あなたの選択肢: "
    text = "[BOS]" + start_sentence
    logger.info(f"Prompt : {text}")
    tokenized_text = tokenizer(text, return_tensors="pt")
    text_id = tokenized_text["input_ids"].to(rank)

    generator = TextGenerator(model, eos_token_id, rank, tokenizer)

    generated_text = generator.generate_text(
        text_id,
        # max_length=config.dataset.max_seq_length,
        max_length=512,
        temperature=0.8,
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
        print("---")
        print(tokenizer.decode(generated_text[0]))
        print("---")

    dist_cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--start_sentence", type=str)
    args = parser.parse_args()
    config = load_config(args.config_path)
    start_sentence = args.start_sentence
    WORLD_SIZE = torch.cuda.device_count()

    mp.spawn(main, args=(WORLD_SIZE, config, start_sentence), nprocs=WORLD_SIZE, join=True)
