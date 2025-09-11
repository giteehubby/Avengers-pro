from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import requests
import time
import random
from openai import OpenAI


# Model Classes
class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
    
    @abstractmethod
    def load(self, **kwargs):
        """Load the model"""
        pass
    
    @abstractmethod
    def inference(self, prompt: str, max_tokens: int = 4096) -> Tuple[str, List[Dict]]:
        """Run inference on the model
        
        Returns:
            Tuple of (response, messages) where messages is the complete conversation history
        """
        pass


class APIModel(BaseModel):
    """
    A model wrapper for OpenAI-compatible API servers.
    
    This class uses the OpenAI client with a custom base URL to connect to
    various API providers that support the OpenAI API format.
    """
    
    def __init__(self, model_name: str, base_url: str, api_key: str, max_retries: int = 5, **kwargs):
        """
        Initializes the APIModel.
        
        Args:
            model_name (str): A name for the model, used for identification/logging.
            base_url (str): The base URL of the API endpoint (e.g., "https://openrouter.ai/api/v1").
            api_key (str): The API key for authentication.
            max_retries (int): Maximum number of retries for failed API calls (default: 3).
            **kwargs: Additional parameters like extra_headers, site_url, site_name.
        """
        super().__init__(model_name=model_name, **kwargs)
        self.base_url = base_url
        self.api_key = api_key
        self.max_retries = max_retries
        
        # Extract optional parameters
        self.extra_headers = kwargs.get('extra_headers', {})
        self.site_url = kwargs.get('site_url', '')
        self.site_name = kwargs.get('site_name', '')
        
        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        
        # print(f"APIModel initialized for model '{self.model_name}' targeting URL: {self.base_url}")

    def load(self, **kwargs):
        """
        Loads the model. For an API model, this performs a health check.
        """
        print(f"'{self.model_name}' is an API-based model. Performing health check...")
        
        try:
            # Test the connection with a simple request
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1,
                extra_headers=self._get_extra_headers()
            )
            print("API server is alive and responding.")
        except Exception as e:
            print(f"Warning: Could not connect to API server at {self.base_url}. Error: {e}")

    def _get_extra_headers(self) -> Dict[str, str]:
        """Get extra headers for the API request"""
        headers = {}
        
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name
            
        # Add any custom extra headers
        headers.update(self.extra_headers)
        
        return headers

    def _should_retry(self, exception: Exception) -> bool:
        """Determine if the exception should trigger a retry"""
        error_str = str(exception).lower()
        
        # Check for connection-related errors
        retry_indicators = [
            'connection',
            'timeout',
            'network',
            'socket',
            'ssl',
            'connectionerror',
            'connectionpool',
            'max retries',
            'temporary failure',
            'service unavailable',
            'rate limit',
            'too many requests'
        ]
        
        return any(indicator in error_str for indicator in retry_indicators)

    def _calculate_delay(self, attempt: int, base_delay: float = 1.0, max_delay: float = 60.0) -> float:
        """Calculate delay for exponential backoff"""
        # Exponential backoff with jitter
        delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
        return delay

    def inference(self, prompt: str, system_prompt: str = None, max_tokens: int = 4096, **kwargs) -> Tuple[str, List[Dict]]:
        """
        Performs inference by calling the OpenAI-compatible API endpoint with retry mechanism.

        Args:
            prompt (str): The main user prompt/question.
            system_prompt (str, optional): A system-level instruction.
            max_tokens (int): Maximum number of tokens to generate.
            **kwargs: Additional parameters like temperature, top_p, etc.

        Returns:
            Tuple[str, List[Dict]]: A tuple containing:
                - The string response from the API.
                - A reconstructed conversation history for compatibility.
        """
        # Construct messages list
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        # print(f"Sending request to {self.base_url} with model '{self.model_name}'")

        # Retry logic with exponential backoff
        for attempt in range(self.max_retries + 1):
            try:
                # Make the API call using OpenAI client
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    extra_headers=self._get_extra_headers(),
                    **kwargs  # Pass through any additional parameters like temperature, top_p, etc.
                )
                
                # Extract the response
                api_response_text = completion.choices[0].message.content
                
                if not api_response_text:
                    print("Warning: API returned a successful status but the response content was empty.")
                
                # Success - return the response
                complete_messages = messages + [{"role": "assistant", "content": api_response_text}]
                return api_response_text, complete_messages
                
            except Exception as e:
                print(f"Error calling API (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                
                # Check if this is the last attempt or if we shouldn't retry
                if attempt == self.max_retries or not self._should_retry(e):
                    print(f"Max retries ({self.max_retries}) reached or non-retryable error. Giving up.")
                    return f"API Error: {e}", messages
                
                # Calculate delay for exponential backoff
                delay = self._calculate_delay(attempt)
                print(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)

        # This should never be reached, but just in case
        return f"API Error: Max retries exceeded", messages
