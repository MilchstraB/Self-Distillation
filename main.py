from string import Template
from typing import Optional
from dataclasses import dataclass, field

import torch
import transformers
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from distil_trainer import DistilTrainer
from distil_config import DistilConfig


@dataclass
class Arguments(DistilConfig):
    model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen2.5-7B-Instruct",
        metadata={"help": "The model name or path."}
    )
    train_path: str = field(
        default="data/tooluse_data/train_data.json", 
        metadata={"help": "Path to the training data."}
    )
    eval_path: str = field(
        default="data/tooluse_data/eval_data.json",
        metadata={"help": "Path to the evaluation data."}
    )


def load_tooluse_dataset(train_path, test_path, seed=42) -> Dataset:
    """Load and prepare tooluse dataset with formatted prompts."""
    train_dataset = Dataset.from_json(train_path)
    test_dataset = Dataset.from_json(test_path)

    def format_example(example):

        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

        return {
            "prompt": [{"role": "user", "content": example['prompt']}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(orig_content=example['prompt'], output_text='\n'.join(example['golden_response']))}],
        }
    
    train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.shuffle(seed=seed)
    return train_dataset, None


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((Arguments))
    args = parser.parse_args_into_dataclasses()[0]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_tooluse_dataset(args.train_path, args.eval_path, args.seed)[0]


    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()