import torch
from transformers import AutoModelForPreTraining, AutoProcessor

REPO = "dartags/DanbotNL-2408-260M"

processor = AutoProcessor.from_pretrained(
    REPO,
    trust_remote_code=True,
    # revision="f992aa6", # optional
)
model = AutoModelForPreTraining.from_pretrained(
    REPO,
    trust_remote_code=True,
    # revision="f992aa6", # optional
    torch_dtype=torch.bfloat16
)

# 翻訳
inputs = processor(
    encoder_text="一人の猫耳の少女が座ってこっちを見ている。",
    decoder_text=processor.decoder_tokenizer.apply_chat_template(
        {
            "aspect_ratio": "tall",
            "rating": "general",
            "length": "very_short",
            "translate_mode": "exact",
        },
        tokenize=False,
    ),
    return_tensors="pt",
)

with torch.inference_mode():
    outputs = model.generate(
        **inputs.to(model.device),
        do_sample=False,
        eos_token_id=processor.decoder_tokenizer.convert_tokens_to_ids(
            "</translation>"
        ),
        return_dict_in_generate=True,
        output_scores=True,
    )

# 入力部分を除いた生成トークン列
generated_ids = outputs.sequences[0, len(inputs.input_ids[0]):]

# 各トークンのスコア（確率）を取得
tag_scores = []
for token_id, score in zip(generated_ids, outputs.scores):
    probs = torch.softmax(score[0], dim=-1)
    token_prob = probs[token_id].item()
    tag = processor.decoder_tokenizer.decode([token_id], skip_special_tokens=True)
    if tag.strip():
        tag_scores.append((tag.strip(), token_prob))

# スコアの高い順にソートして出力
tag_scores.sort(key=lambda x: x[1], reverse=True)

for tag, score in tag_scores:
    print(f"{tag} {score * 100:.1f}")
