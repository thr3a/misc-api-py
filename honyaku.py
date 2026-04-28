import torch
from transformers import AutoModelForPreTraining, AutoProcessor

REPO = "dartags/DanbotNL-2408-260M"

processor = AutoProcessor.from_pretrained(
    REPO,
    trust_remote_code=True,
    revision="f992aa6", # optional 
)
model = AutoModelForPreTraining.from_pretrained(
    REPO,
    trust_remote_code=True,
    revision="f992aa6", # optional
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
    )
translation = ", ".join(
    tag
    for tag in processor.batch_decode(
        outputs[0, len(inputs.input_ids[0]) :],
        skip_special_tokens=True,
    )
    if tag.strip() != ""
)
print("translation:", translation)
# translation: 1girl, solo, looking at viewer, sitting, cat girl
