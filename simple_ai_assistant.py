
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from tools import AVAILABLE_TOOLS, TOOLS

load_dotenv()

base_url = os.getenv("BASE_URL")
api_key = os.getenv("API_KEY")
model = os.getenv("MODEL")

if not all([base_url, api_key, model]):
    raise ValueError(
        "Missing BASE_URL, API_KEY, or MODEL in your .env file."
    )

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
    timeout=120.0,
    max_retries=0,
)

SYSTEM_PROMPT = """
You are a helpful AI assistant.
Give clear and simple answers.

Use the calculate tool for arithmetic calculations.
Use get_current_datetime when asked for the current date or time.
Never invent a tool result. Explain the result returned by the tool.
"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

MAX_TOOL_ROUNDS = 5


def execute_tool(name: str, arguments: str) -> str:
    """Execute only registered tools and return a JSON result."""
    function = AVAILABLE_TOOLS.get(name)

    if function is None:
        return json.dumps({"error": f"Unknown tool: {name}"})

    try:
        parsed_arguments = json.loads(arguments)

        if not isinstance(parsed_arguments, dict):
            raise ValueError("Tool arguments must be a JSON object.")

        result = function(**parsed_arguments)
        return json.dumps(result, allow_nan=False)

    except (ValueError, TypeError, json.JSONDecodeError) as error:
        return json.dumps({"error": str(error)})

    except Exception:
        return json.dumps({"error": "Tool execution failed."})


def get_response(user_input: str) -> str:
    messages.append({"role": "user", "content": user_input})

    try:
        # Allow the model to request tools, then continue until
        # it produces a normal text response.
        for _ in range(MAX_TOOL_ROUNDS):
            response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=256,
        )




            assistant_message = response.choices[0].message

            # Preserve the assistant's tool-call message in history.
            messages.append(
                assistant_message.model_dump(exclude_none=True)
            )

            tool_calls = assistant_message.tool_calls

            if not tool_calls:
                return assistant_message.content or ""

            for tool_call in tool_calls:
                name = tool_call.function.name
                arguments = tool_call.function.arguments

                print(f"[Tool requested: {name}]")

                result = execute_tool(name, arguments)

                print(f"[Tool result: {result}]")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        return "I reached the tool-call limit. Please try a simpler request."

    except Exception:
        # Roll back this turn if the API fails, so incomplete
        # tool-call history is not reused on the next request.
        messages.pop()
        raise


def main():
    print(f"AI Assistant is ready! Model: {model}")
    print("Try: What is 125 * 48?")
    print("Type 'exit' to quit.")

    while True:
        try:
            user_input = input("\nPriyanka: ").strip()

            if user_input.lower() in {
                "exit", "quit", "bye", "goodbye", "see you later"
            }:
                print("Assistant: Goodbye, Priyanka!")
                break

            if not user_input:
                continue

            reply = get_response(user_input)
            print(f"Assistant: {reply}")

        except KeyboardInterrupt:
            print("\nAssistant: Goodbye!")
            break

        except Exception as error:
            print(f"Assistant error: {error}")


if __name__ == "__main__":
    main()
