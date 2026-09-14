"""Independent Hugging Face sanity check for the 30B STX sampler."""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT = (
    "Ian painted all the outer faces of some identical cuboids. He painted "
    "a total of 48 faces. How many cuboids did Ian paint?\nPlease reason step "
    "by step, and put your final answer within \\boxed{}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="Qwen/Qwen3-30B-A3B-Base")
    parser.add_argument("--save-logprobs")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    messages = [{"role": "user", "content": PROMPT}]
    prompts = {
        "raw": PROMPT + "\n",
        "chat-no-think": tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False),
        "chat-think": tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=True),
    }

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0},
        low_cpu_mem_usage=True).eval()
    eot = tokenizer.convert_tokens_to_ids("<|im_end|>")
    for name, prompt in prompts.items():
        encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
        print(f"\n=== {name} ===")
        print(f"prompt_tokens={encoded.input_ids.shape[-1]} prompt={prompt!r}")
        with torch.inference_mode():
            logprobs = model(**encoded).logits[0, -1].float().log_softmax(-1)
        probs = logprobs.exp()
        values, ids = logprobs.topk(1024)
        entropy = -(probs * logprobs).sum().item()
        print(
            f"top1={values[0].exp().item():.9f} "
            f"top128={values[:128].exp().sum().item():.9f} "
            f"top1024={values.exp().sum().item():.9f} "
            f"entropy={entropy:.6f}")
        print("first-token top 10:")
        for value, token_id in zip(values[:10].tolist(), ids[:10].tolist()):
            print(token_id, f"p={math.exp(value):.6f}",
                  repr(tokenizer.decode([token_id])))
        if args.save_logprobs:
            output = Path(args.save_logprobs).with_name(
                f"{Path(args.save_logprobs).stem}-{name}.npy")
            output.parent.mkdir(parents=True, exist_ok=True)
            np.save(output, logprobs.cpu().numpy())

        stop_ids = ([tokenizer.eos_token_id] if name == "raw"
                    else [tokenizer.eos_token_id, eot])
        common = dict(
            max_new_tokens=args.max_new_tokens,
            eos_token_id=stop_ids,
            pad_token_id=tokenizer.pad_token_id,
        )
        with torch.inference_mode():
            greedy = model.generate(**encoded, do_sample=False, **common)
        print("greedy:", repr(tokenizer.decode(
            greedy[0, encoded.input_ids.shape[-1]:],
            skip_special_tokens=False)))

        batch = {key: value.repeat(4, 1) for key, value in encoded.items()}
        torch.manual_seed(7)
        with torch.inference_mode():
            sampled = model.generate(
                **batch, do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                **common)
        for row in range(len(sampled)):
            print(f"sample {row}:", repr(tokenizer.decode(
                sampled[row, encoded.input_ids.shape[-1]:],
                skip_special_tokens=False)))


if __name__ == "__main__":
    main()
