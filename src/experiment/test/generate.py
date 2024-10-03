import torch
from dacite import Config as DaciteConfig
from dacite import from_dict
from omegaconf import OmegaConf
from transformers import AutoTokenizer, set_seed
from xlstm import xLSTMLMModel, xLSTMLMModelConfig

from src.cfg.load_yaml_cfg import load_config
from src.utils import torch_dtype_map


def generate_text(
    model,
    start_sequence,
    eos_token_id,
    max_length=100,
    temperature=1.0,
):
    current_sequence = start_sequence

    for _ in range(max_length):
        with torch.no_grad():
            logits = model(current_sequence)

        # Apply temperature
        output = logits[:, -1, :] / temperature

        # Sample from the output distribution
        probs = torch.softmax(output, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        current_sequence = torch.cat([current_sequence, next_token], dim=1)
        if next_token.item() == eos_token_id:
            print("EOS token is generated.")
            break

    return current_sequence


def main():
    config = load_config("src/cfg/yaml/v4/generate_config.yaml")
    # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.basic.device_ids))
    set_seed(config.basic.seed)

    model_config = from_dict(
        data_class=xLSTMLMModelConfig,
        data=OmegaConf.to_container(config.model, resolve=True),
        config=DaciteConfig(strict=True),
    )

    model = xLSTMLMModel(model_config)
    model.eval()
    model.load_state_dict(torch.load(config.basic.model_weight_path))
    model = model.to(dtype=torch_dtype_map[config.training.weight_precision])
    model = model.to(config.basic.device)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.name)
    tokenizer.add_special_tokens(
        {"pad_token": "[PAD]", "bos_token": "[BOS]", "eos_token": "[EOS]"}
    )
    eos_token_id = tokenizer.eos_token_id

    text = "[BOS]静岡大学は、"
    tokenized_text = tokenizer(text, return_tensors="pt")

    generated_text = generate_text(
        model,
        tokenized_text["input_ids"].to(config.basic.device),
        eos_token_id,
        max_length=200,
        temperature=1.0,
    )
    print(tokenizer.decode(generated_text[0]))


if __name__ == "__main__":
    main()
