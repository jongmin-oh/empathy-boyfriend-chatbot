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

model_id = "j5ng/EEVE-korean-empathy-chat-10.8B"
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
EOT_TOKEN = tokenizer.eos_token

system_prompt = "남친과 여친의 대화에서 남친이 여친의 말에 공감해주고 있다."


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
        self.encounters = encounters
        self.counter = {tuple(stop.tolist()): 0 for stop in stops}

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.equal(input_ids[0, -len(stop):], stop):
                self.counter[tuple(stop.tolist())] += 1
                if self.counter[tuple(stop.tolist())] >= self.encounters:
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
    prompt = get_prompt(messages, eos_token=EOT_TOKEN)
    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        add_special_tokens=False,
        return_token_type_ids=False,
    ).to("cuda")

    stop_words = [EOT_TOKEN]
    stop_words_ids = [tokenizer.encode(stop_word, add_special_tokens=False) for stop_word in stop_words]
    stop_words_ids = [torch.tensor(ids, device='cuda:0') for ids in stop_words_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    streamer = TextIteratorStreamer(
        tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        input_ids=inputs["input_ids"],
        streamer=streamer,
        max_length=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=stopping_criteria,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)
