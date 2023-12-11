def get_prompt(
    message: str, chat_history: list[tuple[str, str]], system_prompt: str
) -> str:
    texts = [f"{system_prompt}\n"]
    # The first user input is _not_ stripped
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f"\n여친: {user_input}\n남친: {response.strip()}")
    message = message.strip() if do_strip else message
    texts.append(f"\n여친: {message}\n남친:")
    return "".join(texts)
