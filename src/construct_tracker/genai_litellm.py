"""Send a prompt to the specified model and return the generated response."""

from typing import Optional

import litellm
from litellm import completion

litellm.drop_params = True  # will ignore parameters you set if they don't belong in a model


def api_request(
    prompt: str,
    model: str = "commmand-nightly",
    api_key: Optional[str] = None,
    temperature: float = 0.1,
    top_p: float = 1.0,
    timeout: int = 45,
    num_retries: int = 2,
    max_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    response_format: Optional[str] = None,
) -> str:
    """Send a prompt to the specified model and return the generated response.

    Args:
        prompt (str): The input prompt to be sent to the model.
        model (str, optional): The model to use for completion. Defaults to "commmand-nightly".
        api_key (Optional[str], optional): API key for authentication. Defaults to None.
        temperature (float, optional): Sampling temperature to control randomness. Defaults to 0.1.
        top_p (float, optional): Nucleus sampling probability threshold. Defaults to 1.0.
        timeout (int, optional): Timeout for the request in seconds. Defaults to 45.
        num_retries (int, optional): Number of retries in case of failure. Defaults to 2.
        max_tokens (Optional[int], optional): Maximum number of tokens in the generated response. Defaults to None.
        seed (Optional[int], optional): Random seed for reproducibility. Defaults to None.
        response_format (Optional[str], optional): The format of the response. Defaults to None.

    Returns:
        str: The generated response from the model.
    """
    # Open AI status: https://status.openai.com/

    messages = [{"content": prompt, "role": "user"}]
    responses = completion(
        model=model,
        messages=messages,
        api_key=api_key,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
        num_retries=num_retries,
        max_tokens=max_tokens,
        seed=seed,
        # response_format=response_format
    )
    response = responses.get("choices")[0].get("message").get("content")  # access response for first message
    return response
