from typing import Iterator

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from utils import get_prompt

model_id = "j5ng/polyglot-ko-empathy-chat-5.8b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

system_prompt = "아래는 연인간의 대화 내용이다."

if torch.cuda.is_available():
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
else:
    model = None
tokenizer = AutoTokenizer.from_pretrained(model_id)


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
    message: str,
    chat_history: list[tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
) -> Iterator[str]:
    prompt = get_prompt(message, chat_history, system_prompt)
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

    # Set pad_token_id to eos_token_id
    generate_kwargs = dict(
        input_ids=inputs["input_ids"],
        max_length=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        num_beams=1,
        stopping_criteria=stopping_criteria,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Generate response using the model
    outputs = model.generate(**generate_kwargs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.replace(prompt + " ", "")


if __name__ == "__main__":
    chat_history = []
    while True:
        user_input = input("나: ")
        if user_input.lower() == "웅":
            break

        response = run(user_input, chat_history, system_prompt)
        print("남친:", response.replace("</끝>", ""))

        # Update chat history with user input and chatbot response
        chat_history.append((user_input, response))
