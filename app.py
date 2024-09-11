import json
import sseclient
import requests
from flask import Flask, request, Response, stream_with_context
import random

app = Flask(__name__)

def generate_random_ip():
    return f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}"

def generate_user_agent():
    os_list = ['Windows NT 10.0', 'Windows NT 6.1', 'Mac OS X 10_15_7', 'Ubuntu', 'Linux x86_64']
    browser_list = ['Chrome', 'Firefox', 'Safari', 'Edge']
    chrome_version = f"{random.randint(70, 126)}.0.{random.randint(1000, 9999)}.{random.randint(100, 999)}"
    firefox_version = f"{random.randint(70, 100)}.0"
    safari_version = f"{random.randint(600, 615)}.{random.randint(1, 9)}.{random.randint(1, 9)}"
    edge_version = f"{random.randint(80, 100)}.0.{random.randint(1000, 9999)}.{random.randint(100, 999)}"

    os = random.choice(os_list)
    browser = random.choice(browser_list)

    if browser == 'Chrome':
        return f"Mozilla/5.0 ({os}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_version} Safari/537.36"
    elif browser == 'Firefox':
        return f"Mozilla/5.0 ({os}; rv:{firefox_version}) Gecko/20100101 Firefox/{firefox_version}"
    elif browser == 'Safari':
        return f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/{safari_version} (KHTML, like Gecko) Version/{safari_version.split('.')[0]}.1.2 Safari/{safari_version}"
    elif browser == 'Edge':
        return f"Mozilla/5.0 ({os}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{edge_version} Safari/537.36 Edg/{edge_version}"

def format_openai_response(content, finish_reason=None):
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "created": 1677652288,
        "model": "gpt-4o",
        "choices": [{
            "delta": {"content": content} if content else {"finish_reason": finish_reason},
            "index": 0,
            "finish_reason": finish_reason
        }]
    }

@app.route('/hf/v1/chat/completions', methods=['POST'])
def chat_completions():
    data = request.json
    messages = data.get('messages', [])
    stream = data.get('stream', False)
    
    if not messages:
        return {"error": "No messages provided"}, 400
    
    model = data.get('model', 'gpt-4o')

    if model.startswith('gpt'):
        endpoint = "openAI"
        original_api_url = 'https://chatpro.ai-pro.org/api/ask/openAI'
    elif model.startswith('claude'):
        endpoint = "claude"
        original_api_url = 'https://chatpro.ai-pro.org/api/ask/claude'
    else:
        return {"error": "Unsupported model"}, 400

    headers = {
        'content-type': 'application/json',
        'X-Forwarded-For': generate_random_ip(),
        'origin': 'https://chatpro.ai-pro.org',
        'user-agent': generate_user_agent()
    }

    def generate():
        nonlocal messages
        full_response = ""
        while True:
            conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            conversation += "\nPlease follow and reply to the user's recent messages and avoid answers that summarize the conversation history."
            
            payload = {
                "text": conversation,
                "endpoint": endpoint,
                "model": model
            }
            
            response = requests.post(original_api_url, headers=headers, json=payload, stream=True)
            client = sseclient.SSEClient(response)
            
            for event in client.events():
                if event.data.startswith('{"text":'):
                    data = json.loads(event.data)
                    new_content = data['text'][len(full_response):]
                    full_response = data['text']
                    
                    if new_content:
                        yield f"data: {json.dumps(format_openai_response(new_content))}\n\n"
                
                elif '"final":true' in event.data:
                    final_data = json.loads(event.data)
                    response_message = final_data.get('responseMessage', {})
                    finish_reason = response_message.get('finish_reason', 'stop')
                    
                    if finish_reason == 'length':
                        messages.append({"role": "assistant", "content": full_response})
                        messages.append({"role": "user", "content": "Please continue your output and do not repeat the previous content"})
                        break  # Jump out of the current loop and continue with the next request
                    else:
                        # End normally, sending the final content (if any)
                        last_content = response_message.get('text', '')
                        if last_content and last_content != full_response:
                            yield f"data: {json.dumps(format_openai_response(last_content[len(full_response):]))}\n\n"
                        
                        yield f"data: {json.dumps(format_openai_response('', finish_reason))}\n\n"
                        yield "data: [DONE]\n\n"
                        return  # completely end generation

        # If it ends due to multiple length limits, send a stop signal
        yield f"data: {json.dumps(format_openai_response('', 'stop'))}\n\n"
        yield "data: [DONE]\n\n"

    if stream:
        return Response(stream_with_context(generate()), content_type='text/event-stream')
    else:
        full_response = ""
        finish_reason = "stop"
        for chunk in generate():
            if chunk.startswith("data: ") and not chunk.strip() == "data: [DONE]":
                response_data = json.loads(chunk[6:])
                if 'choices' in response_data and response_data['choices']:
                    delta = response_data['choices'][0].get('delta', {})
                    if 'content' in delta:
                        full_response += delta['content']
                    if 'finish_reason' in delta:
                        finish_reason = delta['finish_reason']

        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_response
                },
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

if __name__ == '__main__':
    app.run(debug=True, port=5000)
