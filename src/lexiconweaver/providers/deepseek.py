"""DeepSeek LLM provider implementation."""

import asyncio
import json
from typing import AsyncIterator

import httpx

from lexiconweaver.config import DeepSeekConfig
from lexiconweaver.exceptions import ProviderError
from lexiconweaver.logging_config import get_logger
from lexiconweaver.providers.base import BaseLLMProvider

logger = get_logger(__name__)


class DeepSeekProvider(BaseLLMProvider):
    """LLM provider for DeepSeek API (V3)."""

    def __init__(self, config: DeepSeekConfig) -> None:
        """Initialize the DeepSeek provider.
        
        Args:
            config: DeepSeek configuration with api_key, model, base_url, timeout, max_retries.
        """
        self.api_key = config.api_key
        self.model = config.model
        self.base_url = config.base_url.rstrip("/")
        self.timeout = config.timeout
        self.max_retries = config.max_retries

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "deepseek"

    async def is_available(self) -> bool:
        """Check if DeepSeek API is available and configured."""
        if not self.api_key or not self.api_key.strip():
            logger.debug("DeepSeek API key not configured")
            return False

        try:
            # Use a minimal chat completion to verify API key and endpoint (most reliable)
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key.strip()}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Say OK"}],
                "stream": False,
                "max_tokens": 5,
            }
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code == 200:
                    return True
                if response.status_code == 401:
                    logger.debug("DeepSeek API key invalid or expired")
                    return False
                logger.debug(
                    "DeepSeek availability check failed",
                    status=response.status_code,
                    body=response.text[:200],
                )
                return False
        except Exception as e:
            logger.debug("DeepSeek availability check failed", error=str(e))
            return False

    async def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate a response from DeepSeek.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
        
        Returns:
            The generated text response.
        
        Raises:
            ProviderError: If the API call fails after retries.
        """
        api_key = (self.api_key or "").strip()
        if not api_key:
            raise ProviderError(
                "DeepSeek API key not configured. Set LEXICONWEAVER_DEEPSEEK__API_KEY environment variable.",
                provider="deepseek",
            )

        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": 0.6,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload, headers=headers)

                    if response.status_code == 401:
                        raise ProviderError(
                            "DeepSeek API authentication failed. Check your API key.",
                            provider="deepseek",
                        )

                    if response.status_code == 429:
                        # Rate limited - wait and retry
                        if attempt < self.max_retries:
                            wait_time = 2 ** (attempt + 1)  # Exponential backoff
                            logger.warning(
                                "DeepSeek rate limited, retrying",
                                attempt=attempt + 1,
                                wait_time=wait_time,
                            )
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise ProviderError(
                                "DeepSeek API rate limit exceeded after retries",
                                provider="deepseek",
                            )

                    if response.status_code != 200:
                        raise ProviderError(
                            f"DeepSeek API returned status {response.status_code}: {response.text}",
                            provider="deepseek",
                        )

                    result = response.json()
                    choices = result.get("choices", [])
                    if not choices:
                        raise ProviderError(
                            "DeepSeek API returned no choices",
                            provider="deepseek",
                        )

                    return choices[0].get("message", {}).get("content", "").strip()

            except httpx.TimeoutException as e:
                if attempt < self.max_retries:
                    wait_time = 2**attempt
                    logger.warning(
                        "DeepSeek timeout, retrying",
                        attempt=attempt + 1,
                        wait_time=wait_time,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise ProviderError(
                        f"DeepSeek request timed out after {self.max_retries + 1} attempts",
                        provider="deepseek",
                    ) from e

            except httpx.RequestError as e:
                raise ProviderError(
                    f"Failed to connect to DeepSeek API: {e}",
                    provider="deepseek",
                ) from e

            except ProviderError:
                raise

            except Exception as e:
                raise ProviderError(
                    f"Unexpected error calling DeepSeek: {e}",
                    provider="deepseek",
                ) from e

        raise ProviderError(
            "Failed to get response from DeepSeek after retries",
            provider="deepseek",
        )

    async def generate_streaming(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        """Generate a streaming response from DeepSeek.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
        
        Yields:
            Text chunks as they arrive from DeepSeek.
        
        Raises:
            ProviderError: If the API call fails.
        """
        api_key = (self.api_key or "").strip()
        if not api_key:
            raise ProviderError(
                "DeepSeek API key not configured. Set LEXICONWEAVER_DEEPSEEK__API_KEY environment variable.",
                provider="deepseek",
            )

        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": 0.6,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST", url, json=payload, headers=headers
                ) as response:
                    if response.status_code == 401:
                        raise ProviderError(
                            "DeepSeek API authentication failed. Check your API key.",
                            provider="deepseek",
                        )

                    if response.status_code != 200:
                        text = await response.aread()
                        raise ProviderError(
                            f"DeepSeek API returned status {response.status_code}: {text.decode()}",
                            provider="deepseek",
                        )

                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        # SSE format: data: {...}
                        if line.startswith("data: "):
                            data_str = line[6:]  # Remove "data: " prefix

                            if data_str == "[DONE]":
                                break

                            try:
                                data = json.loads(data_str)
                                choices = data.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue

        except httpx.TimeoutException as e:
            raise ProviderError(
                f"DeepSeek streaming request timed out: {e}",
                provider="deepseek",
            ) from e
        except httpx.RequestError as e:
            raise ProviderError(
                f"Failed to connect to DeepSeek API: {e}",
                provider="deepseek",
            ) from e
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(
                f"Unexpected error streaming from DeepSeek: {e}",
                provider="deepseek",
            ) from e
