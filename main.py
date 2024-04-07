import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
import transformers
import json
from tqdm import tqdm

def main():
    model_id = "google-t5/t5-base"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, quantization_config=bnb_config)

    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )

    model = get_peft_model(model, config)
    
    data = load_dataset("json", data_files="dataset.json")
    
    def preprocess_function(examples):
        inputs = [ex["subdomain"] for ex in examples["translation"]]
        targets = [ex["permutation"] for ex in examples["translation"]]
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=128, truncation=True
        )
        return model_inputs
    
    tokenized_datasets = data.map(
        preprocess_function,
        batched=True,
        remove_columns=data["train"].column_names,
    )
    

    trainer = transformers.Seq2SeqTrainer(
        model=model,
        train_dataset=tokenized_datasets['train'],
        args=transformers.Seq2SeqTrainingArguments(
            per_device_train_batch_size=1,
            gradient_checkpointing=False,
            warmup_steps=2,
            remove_unused_columns=False,
            learning_rate=2e-4,
            logging_steps=5,
            output_dir="outputs"
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, model=model)
    )
    
    model.config.use_cache = False
    
    trainer.train()
    
    model.save_pretrained("trained-model")
    
main()
