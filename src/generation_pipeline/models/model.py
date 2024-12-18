import os
from dotenv import load_dotenv
import yaml
from logger import logging

# Load environment variables
load_dotenv()


class ModelProvider:
    def __init__(self, config_path: str):
        """
        Initialize the ModelProvider with configurations.

        :param config_path: Path to the YAML configuration file.
        """
        self.config = self._load_config(config_path)
        self.default_provider = self.config.get("default_provider")
        self.current_provider = None
        self.current_model = None
        logging.info("ModelProvider initialized.")

        # Automatically load the default provider
        self.switch_provider(self.default_provider)

    def _load_config(self, path: str) -> dict:
        """
        Load the configuration file.

        :param path: Path to the YAML file.
        :return: Parsed configuration dictionary.
        """
        with open(path, "r") as file:
            return yaml.safe_load(file)

    def switch_provider(self, provider_name: str):
        """
        Switch the active provider and load its corresponding model.
        :param provider_name: Name of the provider to switch to.
        :return: Loaded model instance.
        """
        if provider_name not in self.config["providers"]:
            raise ValueError(f"Provider '{provider_name}' not found in configuration.")

        logging.info(f"Switching provider to '{provider_name}'...")
        provider_config = self.config["providers"][provider_name]

        actual_provider = provider_config.get("provider_name", provider_name)  # Map to base provider
        if actual_provider == "openai":
            self.current_model = self._load_openai(provider_config)
        elif actual_provider == "anthropic":
            self.current_model = self._load_anthropic(provider_config)
        elif actual_provider == "gemini":
            self.current_model = self._load_gemini(provider_config)
        elif actual_provider == "ollama":
            self.current_model = self._load_ollama(provider_config)
        else:
            raise ValueError(f"Unsupported provider: {actual_provider}")

        self.current_provider = provider_name
        logging.info(f"Successfully switched to provider '{provider_name}'.")
        return self.current_model


    def _load_openai(self, config):
        """Load OpenAI model."""
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment variables.")
        logging.info("Loading OpenAI model...")
        return ChatOpenAI(model=config["model"], api_key=api_key)

    def _load_anthropic(self, config):
        """Load Anthropic model."""
        from langchain_anthropic import ChatAnthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment variables.")
        logging.info("Loading Anthropic model...")
        return ChatAnthropic(model=config["model"], api_key=api_key)

    def _load_gemini(self, config):
        """Load Google Gemini model."""
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment variables.")
        logging.info("Loading Google Gemini model...")
        return ChatGoogleGenerativeAI(model=config["model"], api_key=api_key)

    def _load_ollama(self, config):
        """Load Ollama configuration."""
        logging.info("Loading Ollama configuration...")
        return {"name": config["model"], "endpoint": config["endpoint"]}

    def _load_cohere(self, config):
        """Load Cohere model."""
        from cohere import Client
        api_key = os.getenv(config["api_key_env"])
        if not api_key:
            raise ValueError(f"{config['api_key_env']} not set in environment variables.")
        logging.info("Loading Cohere model...")
        return Client(api_key=api_key)
    

if __name__ == "__main__":
    # Initialize the ModelProvider with configuration
    model_provider = ModelProvider(config_path="config\config.yaml")

    # Default provider (OpenAI)
    print(f"Using default provider: {model_provider.current_provider}")
    openai_model = model_provider.current_model

    # Switch to OpenAI
    anthropic_model = model_provider.switch_provider("openai_gpt35")
    print(f"Switched to provider: {model_provider.current_provider}")

    # Switch to Anthropic
    anthropic_model = model_provider.switch_provider("anthropic")
    print(f"Switched to provider: {model_provider.current_provider}")

    # Switch to Ollama
    ollama_model = model_provider.switch_provider("ollama")
    print(f"Switched to provider: {model_provider.current_provider}")
