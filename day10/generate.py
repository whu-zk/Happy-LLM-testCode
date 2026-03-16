import os
import json
import argparse
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F
import sentencepiece as spm

from model import Llama, ModelArgs


class TextGenerator:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        config_path: Optional[str] = None,
        device: str = "cuda",
    ):
        self.device = device
        
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            model_args = ModelArgs(**config["model"])
        else:
            model_args = ModelArgs()
        
        self.model = Llama(model_args).to(device)
        
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_path)
        
        self.max_seq_len = model_args.max_seq_len
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
    ) -> str:
        tokens = self.tokenizer.encode(prompt)
        tokens = torch.tensor([tokens], dtype=torch.long, device=self.device)
        
        generated_tokens = []
        
        for _ in range(max_new_tokens):
            if tokens.shape[1] > self.max_seq_len:
                context = tokens[:, -self.max_seq_len:]
            else:
                context = tokens
            
            logits = self.model(context)
            logits = logits[:, -1, :]
            
            if repetition_penalty > 1.0:
                for token in tokens[0].tolist():
                    logits[0, token] /= repetition_penalty
            
            logits = logits / temperature
            
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")
            
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            tokens = torch.cat([tokens, next_token], dim=1)
            generated_tokens.append(next_token.item())
            
            if next_token.item() == self.tokenizer.eos_id():
                break
        
        output_tokens = tokens[0].tolist()
        output_text = self.tokenizer.decode(output_tokens)
        
        return output_text
    
    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
    ):
        tokens = self.tokenizer.encode(prompt)
        tokens = torch.tensor([tokens], dtype=torch.long, device=self.device)
        
        yield prompt
        
        for _ in range(max_new_tokens):
            if tokens.shape[1] > self.max_seq_len:
                context = tokens[:, -self.max_seq_len:]
            else:
                context = tokens
            
            logits = self.model(context)
            logits = logits[:, -1, :]
            
            if repetition_penalty > 1.0:
                for token in tokens[0].tolist():
                    logits[0, token] /= repetition_penalty
            
            logits = logits / temperature
            
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")
            
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            tokens = torch.cat([tokens, next_token], dim=1)
            
            next_text = self.tokenizer.decode([next_token.item()])
            yield next_text
            
            if next_token.item() == self.tokenizer.eos_id():
                break
    
    def chat(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
    ):
        print(f"User: {prompt}")
        print("Assistant: ", end="")
        
        full_response = ""
        for text in self.generate_stream(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        ):
            print(text, end="", flush=True)
            full_response = text
        
        print("\n")
        return full_response


def demo_sampling_strategies(generator: TextGenerator, prompt: str):
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print("=" * 60)
    
    print("\n--- Greedy Decoding (temperature=0) ---")
    output = generator.generate(prompt, max_new_tokens=50, temperature=1.0, do_sample=False)
    print(output)
    
    print("\n--- Low Temperature (0.5) ---")
    output = generator.generate(prompt, max_new_tokens=50, temperature=0.5, do_sample=True)
    print(output)
    
    print("\n--- High Temperature (1.5) ---")
    output = generator.generate(prompt, max_new_tokens=50, temperature=1.5, do_sample=True)
    print(output)
    
    print("\n--- Top-K Sampling (k=10) ---")
    output = generator.generate(prompt, max_new_tokens=50, temperature=1.0, top_k=10, do_sample=True)
    print(output)
    
    print("\n--- Top-P (Nucleus) Sampling (p=0.9) ---")
    output = generator.generate(prompt, max_new_tokens=50, temperature=1.0, top_p=0.9, do_sample=True)
    print(output)
    
    print("\n--- Combined: Top-K=40 + Top-P=0.9 ---")
    output = generator.generate(
        prompt,
        max_new_tokens=50,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        do_sample=True,
    )
    print(output)


def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained LLaMA model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="./tokenizer.model",
        help="Path to tokenizer model",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="tiny_config.json",
        help="Path to config file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time, there was a little girl named",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-K sampling parameter",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top-P (nucleus) sampling parameter",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Repetition penalty",
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling instead of greedy decoding",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with different sampling strategies",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive chat mode",
    )
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    generator = TextGenerator(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        config_path=args.config_path,
        device=device,
    )
    
    if args.demo:
        demo_sampling_strategies(generator, args.prompt)
    elif args.interactive:
        print("Interactive Chat Mode (type 'quit' to exit)")
        print("=" * 60)
        while True:
            prompt = input("\nUser: ")
            if prompt.lower() == "quit":
                break
            generator.chat(
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k or 40,
                top_p=args.top_p or 0.9,
            )
    else:
        print(f"Prompt: {args.prompt}")
        print("-" * 60)
        
        output = generator.generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=args.do_sample,
        )
        
        print(output)


if __name__ == "__main__":
    main()
