import os
from threading import Thread
from typing import Iterator

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from utils import get_prompt

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_id = "j5ng/polyglot-ko-empathy-chat-5.8b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
else:
    model = None

tokenizer = AutoTokenizer.from_pretrained(model_id)

system_prompt = "아래는 연인간의 대화 내용이다."


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True

        return False


def run(
    messages: list[dict[str]],
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
) -> Iterator[str]:
    prompt = get_prompt(messages)
    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        add_special_tokens=False,
        return_token_type_ids=False,
    ).to("cuda")

    stop_words = ["</끝>"]
    stop_words_ids = [
        tokenizer(stop_word, return_tensors="pt").to("cuda")["input_ids"].squeeze()
        for stop_word in stop_words
    ]
    stopping_criteria = StoppingCriteriaList(
        [StoppingCriteriaSub(stops=stop_words_ids)]
    )

    streamer = TextIteratorStreamer(
        tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        num_beams=1,
        stopping_criteria=stopping_criteria,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)
