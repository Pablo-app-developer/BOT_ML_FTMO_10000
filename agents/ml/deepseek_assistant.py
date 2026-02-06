"""
DeepSeek API Helper - Asistente de análisis para el bot FTMO
"""
import os
from dotenv import load_dotenv
import requests
import json

# Load environment variables
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


class DeepSeekAssistant:
    """
    Helper class for DeepSeek API interactions
    Use for: log analysis, strategy suggestions, debugging help
    """
    
    def __init__(self):
        if not DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY not found in .env file")
        
        self.api_key = DEEPSEEK_API_KEY
        self.base_url = DEEPSEEK_BASE_URL
        self.model = DEEPSEEK_MODEL
    
    def chat(self, messages: list, temperature: float = 0.7, max_tokens: int = 2000):
        """
        Send chat request to DeepSeek
        
        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            temperature: 0.0-1.0 (0 = deterministic, 1 = creative)
            max_tokens: Max response length
        
        Returns:
            Response text
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
        
        except requests.exceptions.RequestException as e:
            print(f"❌ DeepSeek API Error: {e}")
            return None
    
    def analyze_logs(self, log_file: str):
        """
        Analyze trading logs for issues
        """
        try:
            with open(log_file, 'r') as f:
                logs = f.read()[-5000:]  # Last 5000 chars
        except FileNotFoundError:
            return "Log file not found"
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert trading bot analyst. Identify errors, anomalies, and improvement suggestions."
            },
            {
                "role": "user",
                "content": f"Analyze these trading bot logs:\n\n{logs}"
            }
        ]
        
        return self.chat(messages, temperature=0.3)
    
    def suggest_strategy(self, market_data: dict):
        """
        Get strategy suggestions based on market conditions
        """
        messages = [
            {
                "role": "system",
                "content": "You are a quant trading strategist specializing in forex EUR/USD and FTMO challenges."
            },
            {
                "role": "user",
                "content": f"Given this market data:\n{json.dumps(market_data, indent=2)}\n\nSuggest optimal entry/exit strategy for FTMO 10k challenge."
            }
        ]
        
        return self.chat(messages, temperature=0.5)
    
    def debug_code(self, error_message: str, code_snippet: str):
        """
        Get debugging help for code issues
        """
        messages = [
            {
                "role": "system",
                "content": "You are a Python expert specializing in trading bots and machine learning."
            },
            {
                "role": "user",
                "content": f"Error:\n{error_message}\n\nCode:\n```python\n{code_snippet}\n```\n\nHow do I fix this?"
            }
        ]
        
        return self.chat(messages, temperature=0.3)


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("🤖 DeepSeek Assistant - FTMO Bot Helper")
    print("="*60)
    
    try:
        assistant = DeepSeekAssistant()
        print("✅ DeepSeek API connected successfully!")
        
        # Test simple query
        messages = [
            {
                "role": "system",
                "content": "You are a helpful FTMO trading assistant."
            },
            {
                "role": "user",
                "content": "What are the 3 most important risk management rules for FTMO 10k challenge?"
            }
        ]
        
        print("\n📊 Testing API with sample question...")
        response = assistant.chat(messages, temperature=0.7)
        
        if response:
            print("\n💬 DeepSeek Response:")
            print("-" * 60)
            print(response)
            print("-" * 60)
            print("\n✅ Test successful! API is working.")
        else:
            print("\n❌ No response received")
    
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure .env file exists with DEEPSEEK_API_KEY")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
