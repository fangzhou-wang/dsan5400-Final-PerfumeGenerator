# from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
# from datasets import load_dataset

# def fine_tune_gpt2(train_file, output_dir):
#     """
#     微调 GPT-2 模型
    
#     Args:
#         train_file (str): 训练数据文件 (JSONL 格式)。
#         output_dir (str): 微调模型的保存路径。
#     """
#     # 加载数据集
#     dataset = load_dataset("json", data_files={"train": train_file}, split="train")
    
#     # 加载 GPT-2 模型和分词器
#     model_name = "gpt2"
#     tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#     model = GPT2LMHeadModel.from_pretrained(model_name)
#     tokenizer.pad_token = tokenizer.eos_token  # 添加 padding token

#     # 数据整理器
#     data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

#     # 训练参数
#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         per_device_train_batch_size=2,
#         num_train_epochs=5,
#         logging_steps=100,
#         save_steps=500,
#         save_total_limit=2,
#         learning_rate=5e-5,
#         logging_dir="./train_logs",
#     )

#     # 创建 Trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=dataset,
#         data_collator=data_collator,
#     )

#     # 训练模型
#     trainer.train()
#     model.save_pretrained(output_dir)
#     tokenizer.save_pretrained(output_dir)
#     print(f"Model fine-tuned and saved to {output_dir}")

# # 运行微调
# if __name__ == "__main__":
#     fine_tune_gpt2("../../data/perfume_reviews.jsonl", "../../../fine_tuned_gpt2_perfume")








from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import os

# Word segmentation and annotation functions
def tokenize_function(batch):
    prompts = batch["prompt"]
    completions = batch["completion"]

    # Combine and divide words
    full_texts = [prompt + " " + completion for prompt, completion in zip(prompts, completions)]
    tokenized_examples = tokenizer(
        full_texts,
        max_length=512,  # Limit maximum length
        padding="max_length",  # Automatically fill to maximum length
        truncation=True,  # Enable truncation
    )

    # Set label as input ID
    tokenized_examples["labels"] = tokenized_examples["input_ids"].copy()
    return tokenized_examples


# Fine tuning function
def fine_tune_gpt2(train_file, output_dir):
    # Load data set
    dataset = load_dataset("json", data_files={"train": train_file}, split="train")

    # Load GPT-2 model and word divider
    model_name = "gpt2"
    global tokenizer  
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Set the fill token
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))

    # Word segmentation and labeling
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    # Configure the data organizer
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # The mask language model is not enabled
    )

    # Training parameter
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="no", 
        per_device_train_batch_size=4,  
        num_train_epochs=3,  
        save_strategy="epoch",  
        logging_dir="./logs",  
        logging_steps=50,
        save_total_limit=2,  
        learning_rate=5e-5,
        remove_unused_columns=False, 
    )

    # set Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # start training
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model fine-tuned and saved to {output_dir}")


if __name__ == "__main__":
    fine_tune_gpt2("../../data/perfume_reviews.jsonl", "../../../fine_tuned_gpt2_perfume")
