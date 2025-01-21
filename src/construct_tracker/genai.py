"""Send a prompt to the specified model and return the generated response."""

import json
import os
from typing import Dict, Optional, Tuple

import requests

# see genai_litellm for alternative approach.


def api_request(
    prompt: str,
    model: str = "meta-llama/llama-3.1-405b-instruct:free",
    temperature: float = 0.1,
    top_p: float = 1.0,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[Dict, Dict]:
    """Send a prompt to the specified model and return the generated response.

    Args:
          prompt (str): The input prompt to be sent to the model.
          model (str, optional): The model to use for completion. Defaults to 'meta-llama/llama-3.1-405b-instruct:free'. free models: certain amount of requests per minute per day. See https://openrouter.ai/docs/limits
          api_key (Optional[str], optional): API key for authentication. Defaults to None.
          temperature (float, optional): Sampling temperature to control randomness. Defaults to 0.1.
          top_p (float, optional): Nucleus sampling probability threshold. Defaults to 1.0.
          max_tokens (Optional[int], optional): Maximum number of tokens in the generated response. Defaults to None.
          seed (Optional[int], optional): Random seed for reproducibility. Defaults to None.
          response_format (Optional[dict], optional): The format of the response. Defaults to {'type': 'json_object'}.

    Returns:
          str: The generated response from the model.

    """
    api_key = os.environ.get("api_key")
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
        },
        data=json.dumps(
            {
                "model": model,  # Optional
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "seed": seed,
                "messages": [{"role": "user", "content": prompt}],
            }
        ),
    )
    metadata = json.loads(response.text.strip())

    try:
        final_result = metadata["choices"][0]["message"]["content"]
        # final_result = dict(eval(final_result))
        return final_result, metadata
    except Exception as e:
        print("Error:", e)
        print("""
            Here are some common errors:
            - some errors described here: https://openrouter.ai/docs/errors
            - {'error': {'message': 'No auth credentials found', 'code': 401}} means the The key is invalid
            - did not follow the instructions well. gpt-4o,gpt-4o-mini, gemini 1.5, claude 3.5  work well.
            - you did not provide an API key
            - Rate limit exceeded for free models
            - You do not have funds associated with your API key

            """)
        print("Here's the complete metadata. construct-tracker wishes to parse the string in 'content':")
        print(metadata)

        return


def process_api_output(output_str: str) -> dict:
    r"""
    Processes the API output string and returns a dictionary.

    Args:
      output_str: The string output from the API call.

    Returns:
      A dictionary containing the extracted data.

    Example:
      output_str1 = '{  "desire to escape": [[1], ["I want out"]],  "loneliness": [[1], ["No one cares about me"]],  "suicidal ideation": [[0.5], ["I want out", "It won't get better"]] }'
      output_str2 = '{  "desire to escape": [[1], ["I want out"]],  "loneliness": [[1], ["No one cares about me"]],  "suicidal ideation": [[0.5], ["I want out", "It won't get better"]] }Explanation: - The text clearly expresses a "desire to escape" with the phrase "I want out", which suggests a strong desire to leave the current situation.- The text also clearly expresses "loneliness" with the phrase "No one cares about me", which indicates feelings of isolation and disconnection.- The text may suggest "suicidal ideation" with the phrases "I want out" and "It won't get better", but it\'s not explicitly stated, hence the lower score.'
      output_str3 = '{  "desire to escape": [[1], ["I want out"]],  "loneliness": [[1], ["No one cares about me"]],  "suicidal ideation": [[0.5], ["I want out", "It won't get better"]] }Some additional information here.'

      print(process_api_output(output_str1))
      print(process_api_output(output_str2))
      print(process_api_output(output_str3))
    """
    data = {}
    start_index = output_str.find("{")
    end_index = output_str.rfind("}") + 1

    try:
        # Attempt to directly load the JSON string
        data = json.loads(output_str)
    except json.JSONDecodeError:
        # If JSON decoding fails, try to extract the JSON part
        if start_index != -1 and end_index != -1:
            json_part = output_str[start_index:end_index]
            data = json.loads(json_part)
            return
        else:
            raise ValueError("Invalid API output format.")

    # Extract the additional note if it exists
    if start_index != 0 or end_index != len(output_str):
        data["Additional note"] = output_str[end_index:].strip()

    return data
