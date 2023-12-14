system_prompt = "아래는 연인간의 대화 내용이다."


def get_prompt(messages):
    instruction = f"{system_prompt}\n"
    conversation = []

    for entry in messages:
        role = entry["role"]
        content = entry["content"]

        if role == "user":
            conversation.append(f"여친: {content}")
        elif role == "assistant":
            conversation.append(f"남친: {content}</끝>")

    return instruction + "\n".join(conversation) + "\n남친:"
