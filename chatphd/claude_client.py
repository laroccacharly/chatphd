import anthropic

client = None 

def get_client() -> anthropic.Anthropic:
    global client
    if client is None:
        client = anthropic.Anthropic()
    return client
