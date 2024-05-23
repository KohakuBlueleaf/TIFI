import torch

from transformers import CLIPTokenizer, CLIPTextModel


def load_tokenizers(model="stabilityai/stable-diffusion-xl-base-1.0"):
    tokenizer = CLIPTokenizer.from_pretrained(model, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(model, subfolder="tokenizer_2")
    return tokenizer, tokenizer_2


def encode_prompts_single(
    tokenizer: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    text_encoder_2: CLIPTextModel,
    prompt: str,
):
    max_length = tokenizer.model_max_length

    input_ids = tokenizer(
        prompt, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids.to("cuda")
    input_ids2 = tokenizer_2(
        prompt, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids.to("cuda")

    concat_embeds = []
    for i in range(0, input_ids.shape[-1], max_length):
        concat_embeds.append(text_encoder(input_ids[:, i : i + max_length])[0])

    concat_embeds2 = []
    pooled_embeds2 = []
    for i in range(0, input_ids.shape[-1], max_length):
        hidden_states = text_encoder_2(
            input_ids2[:, i : i + max_length], output_hidden_states=True
        )
        concat_embeds2.append(hidden_states.hidden_states[-2])
        pooled_embeds2.append(hidden_states[0])

    prompt_embeds = torch.cat(concat_embeds, dim=1)
    prompt_embeds2 = torch.cat(concat_embeds2, dim=1)
    prompt_embeds = torch.cat([prompt_embeds, prompt_embeds2], dim=-1)
    pooled_embeds2 = torch.mean(torch.stack(pooled_embeds2, dim=0), dim=0)

    return prompt_embeds, pooled_embeds2


def encode_prompts(
    tokenizer: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    text_encoder_2: CLIPTextModel,
    prompt: str,
    neg_prompt: str,
):
    max_length = tokenizer.model_max_length

    input_ids = tokenizer(
        prompt, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids.to("cuda")
    input_ids2 = tokenizer_2(
        prompt, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids.to("cuda")

    negative_ids = tokenizer(
        neg_prompt,
        truncation=False,
        padding="max_length",
        max_length=input_ids.shape[-1],
        return_tensors="pt",
    ).input_ids.to("cuda")
    negative_ids2 = tokenizer_2(
        neg_prompt,
        truncation=False,
        padding="max_length",
        max_length=input_ids.shape[-1],
        return_tensors="pt",
    ).input_ids.to("cuda")

    if negative_ids.size() > input_ids.size():
        input_ids = tokenizer(
            prompt,
            truncation=False,
            padding="max_length",
            max_length=negative_ids.shape[-1],
            return_tensors="pt",
        ).input_ids.to("cuda")
        input_ids2 = tokenizer_2(
            prompt,
            truncation=False,
            padding="max_length",
            max_length=negative_ids.shape[-1],
            return_tensors="pt",
        ).input_ids.to("cuda")

    concat_embeds = []
    neg_embeds = []
    for i in range(0, input_ids.shape[-1], max_length):
        concat_embeds.append(text_encoder(input_ids[:, i : i + max_length])[0])
        neg_embeds.append(text_encoder(negative_ids[:, i : i + max_length])[0])

    concat_embeds2 = []
    neg_embeds2 = []
    pooled_embeds2 = []
    neg_pooled_embeds2 = []
    for i in range(0, input_ids.shape[-1], max_length):
        hidden_states = text_encoder_2(
            input_ids2[:, i : i + max_length], output_hidden_states=True
        )
        concat_embeds2.append(hidden_states.hidden_states[-2])
        pooled_embeds2.append(hidden_states[0])

        hidden_states = text_encoder_2(
            negative_ids2[:, i : i + max_length], output_hidden_states=True
        )
        neg_embeds2.append(hidden_states.hidden_states[-2])
        neg_pooled_embeds2.append(hidden_states[0])

    prompt_embeds = torch.cat(concat_embeds, dim=1)
    negative_prompt_embeds = torch.cat(neg_embeds, dim=1)
    prompt_embeds2 = torch.cat(concat_embeds2, dim=1)
    negative_prompt_embeds2 = torch.cat(neg_embeds2, dim=1)
    prompt_embeds = torch.cat([prompt_embeds, prompt_embeds2], dim=-1)
    negative_prompt_embeds = torch.cat(
        [negative_prompt_embeds, negative_prompt_embeds2], dim=-1
    )

    pooled_embeds2 = torch.mean(torch.stack(pooled_embeds2, dim=0), dim=0)
    neg_pooled_embeds2 = torch.mean(torch.stack(neg_pooled_embeds2, dim=0), dim=0)

    return prompt_embeds, negative_prompt_embeds, pooled_embeds2, neg_pooled_embeds2
