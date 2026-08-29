from core.brain import chat as brain_chat
def gateway_chat(user_id,message,source="web"):
    return brain_chat(user_id or "web", message)
