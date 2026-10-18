
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from tools import AVAILABLE_TOOLS, TOOLS
from memory import init_db, save_exchange, load_history, clear_history

load_dotenv()

base_url = os.getenv("BASE_URL")
api_key = os.getenv("API_KEY")
model = os.getenv("MODEL")

if not all([base_url, api_key, model]):
    raise ValueError("Check BASE_URL, API_KEY, and MODEL in .env")

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
    timeout=120.0,
    max_retries=0,
)

SYSTEM_PROMPT = """
You are a helpful AI assistant.
Give clear, simple, and accurate answers.
Use calculate for arithmetic calculations.
Use get_current_datetime for the current date and time.
Never invent a tool result.
"""

init_db()

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    *load_history(),
]

MAX_TOOL_ROUNDS = 3


def execute_tool(name: str, arguments: str) -> str:
    """Execute only registered tools."""
    function = AVAILABLE_TOOLS.get(name)

    if function is None:
        return json.dumps({"error": f"Unknown tool: {name}"})

    try:
        parsed = json.loads(arguments)

        if not isinstance(parsed, dict):
            return json.dumps({"error": "Arguments must be a JSON object."})

        result = function(**parsed)
        return json.dumps(result, allow_nan=False)

    except (ValueError, TypeError, json.JSONDecodeError):
        return json.dumps({"error": "Invalid tool arguments."})
    except Exception:
        return json.dumps({"error": "Tool execution failed."})


def get_response(user_input: str) -> str:
    global messages

    messages.append({"role": "user", "content": user_input})

    try:
        for _ in range(MAX_TOOL_ROUNDS):
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=128,
            )

            assistant_message = response.choices[0].message
            tool_calls = assistant_message.tool_calls

            if not tool_calls:
                reply = assistant_message.content or ""
                save_exchange(user_input, reply)

                # Reload clean, persistent chat history.
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *load_history(),
                ]
                return reply

            messages.append(
                assistant_message.model_dump(exclude_none=True)
            )

            for tool_call in tool_calls:
                name = tool_call.function.name
                arguments = tool_call.function.arguments

                print(f"[Calling tool: {name}]")
                result = execute_tool(name, arguments)
                print(f"[Tool result: {result}]")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        raise RuntimeError("Tool-call limit reached.")

    except Exception:
        # Discard incomplete tool-call history after an error.
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *load_history(),
        ]
        raise


def main():
    print(f"AI Assistant ready! Model: {model}")
    print("Commands: /history, /reset, exit")

    while True:
        try:
            user_input = input("\nPriyanka: ").strip()

            if user_input.lower() in {"exit", "quit", "bye"}:
                print("Assistant: Goodbye!")
                break

            if user_input == "/history":
                history = load_history()

                if not history:
                    print("Assistant: No saved conversation yet.")
                else:
                    for item in history:
                        print(f"{item['role']}: {item['content']}")

                continue

            if user_input == "/reset":
                clear_history()
                messages.clear()
                messages.append({
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                })
                print("Assistant: Conversation history cleared.")
                continue

            if not user_input:
                continue

            print("Assistant:", get_response(user_input))

        except KeyboardInterrupt:
            print("\nAssistant: Goodbye!")
            break

        except Exception as error:
            print(f"Assistant error: {error}")


if __name__ == "__main__":
    main()
