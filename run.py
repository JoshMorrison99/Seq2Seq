from peft import PeftConfig, PeftModel
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


def main():
    model_id = "trained-model"
    
    config = PeftConfig.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config.base_model_name_or_path,
        return_dict=True,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = PeftModel.from_pretrained(model, model_id)
    
    text = "advertise.aol.ca"
    outputs = model.generate(**tokenizer(text, return_tensors="pt").to("cuda"), num_beams=5, max_length=150,)

    decoded_outputs = tokenizer.batch_decode(outputs)
    for idx, output in enumerate(decoded_outputs):
        print(f"Generated sequence {idx + 1}: {output}")
    
main()




