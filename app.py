import time
import random
from pathlib import Path

import streamlit as st
import torch
import numpy as np

from model import run

BASE_DIR = Path(__file__).resolve().parent


def seed_everything(seed):
    torch.manual_seed(seed)  # torch를 거치는 모든 난수들의 생성순서를 고정한다
    torch.cuda.manual_seed(seed)  # cuda를 사용하는 메소드들의 난수시드는 따로 고정해줘야한다
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True  # 딥러닝에 특화된 CuDNN의 난수시드도 고정
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)  # numpy를 사용할 경우 고정
    random.seed(seed)  # 파이썬 자체 모듈 random 모듈의 시드 고정


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] != "system":
            st.markdown(message["content"])

if prompt := st.chat_input("남친에게 하고싶은 말을 입력하세요."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        with st.spinner("답변 생성 중...."):
            time.sleep(random.uniform(1.8, 2.2))

        seed_everything(42)
        stream = run(
            messages=st.session_state.messages,
            max_new_tokens=1024,
            temperature=0.3,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.2,
        )

        for response in stream:
            full_response = response.replace("</끝>", "")
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
