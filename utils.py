system_prompt = "아래는 연인간의 대화 내용이다."


def get_prompt(messages):
    instrution = system_prompt + "\n"
    contexts = "여친: " + messages[0]["content"] + "\n"

    for entry in messages[1:]:
        role = entry["role"]
        content = entry["content"]

        if role == "user":
            contexts += "여친: " + content + "\n"
        elif role == "assistant":
            contexts += "남친:" + content + "</끝>" + "\n"

    return instrution + contexts + "남친:"
