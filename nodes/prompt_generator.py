import os
import requests

class LTX23_KeyframePromptGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "first_frame_description": ("STRING", {"default": "", "multiline": True}),
                "action_flow": ("STRING", {"default": "", "multiline": True}),
                "engine_type": (["OpenAI", "Local API"], {"default": "OpenAI"}),
                "model_name": ("STRING", {"default": "gpt-4o-mini"}),
                "api_url": ("STRING", {"default": "https://api.openai.com/v1"}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ltx_prompt",)
    FUNCTION = "build_ltx_prompt"
    CATEGORY = "LTX2.3/Prompting"

    def build_ltx_prompt(self, first_frame_description, action_flow, engine_type, model_name, api_url, api_key):
        # Resolve credential overrides
        final_key = api_key.strip() if api_key.strip() else os.environ.get("OPENAI_API_KEY", "")
        
        system_instruction = (
            "You are an expert AI video prompt director specialized in LTX 2.3. Synthesize the provided "
            "Initial Frame Condition and Chronological Action Flow into a unified, descriptive LTX 2.3 text block. "
            "Structure: 1. Subject & Starting Textures, 2. Spatial Blocking & Motion Evolution, 3. Camera Choreography, "
            "4. Multi-Layered Audio Design. Respond with ONLY the final prompt text without conversational wrapper text."
        )

        user_content = (
            f"INITIAL FRAME CONDITION:\n{first_frame_description}\n\n"
            f"CHRONOLOGICAL ACTION FLOW:\n{action_flow}"
        )

        # Handling payload construction for OpenAI API standard endpoints
        if engine_type == "OpenAI":
            headers = {"Authorization": f"Bearer {final_key}", "Content-Type": "application/json"}
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content}
                ]
            }
            # Handle common openai url mistake
            base_url = api_url.rstrip('/')
            if base_url.endswith("/v1"):
                endpoint = f"{base_url}/chat/completions"
            else:
                endpoint = f"{base_url}/v1/chat/completions"
            
            try:
                response = requests.post(endpoint, headers=headers, json=payload, timeout=30)
                if response.status_code != 200:
                    return (f"API Error ({response.status_code}): {response.text}",)
                return (response.json()["choices"][0]["message"]["content"].strip(),)
            except Exception as e:
                return (f"API Error: {str(e)}",)
                
        # Handling payload construction for Local Ollama / legacy generation API architectures
        else:
            payload = {
                "model": model_name,
                "prompt": f"{system_instruction}\n\n{user_content}",
                "stream": False
            }
            endpoint = f"{api_url.rstrip('/')}/api/generate"
            
            try:
                response = requests.post(endpoint, json=payload, timeout=45)
                if response.status_code != 200:
                    return (f"Local API Error ({response.status_code}): {response.text}",)
                return (response.json().get("response", "").strip(),)
            except Exception as e:
                return (f"Local API Error: {str(e)}",)

NODE_CLASS_MAPPINGS = {"LTX23_KeyframePromptGenerator": LTX23_KeyframePromptGenerator}
NODE_DISPLAY_NAME_MAPPINGS = {"LTX23_KeyframePromptGenerator": "LTX 2.3 Keyframe & Flow Generator"}
