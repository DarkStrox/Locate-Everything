import requests
import base64
import io
import time
from PIL import Image
from typing import Tuple, List, Dict, Any, Optional
from utils.parser import parse_grounding_response, DetectionBox
from utils.drawer import draw_annotations

DEFAULT_SERVER_URL = "http://127.0.0.1:8085"
DEFAULT_SYSTEM_PROMPT = "You are a visual grounding AI model. Detect and locate objects specified by the user prompt. Return bounding boxes in normalized coordinates format: <ref>object</ref><box><ymin><xmin><ymax><xmax></box>."

class LlamaServerClient:
    @staticmethod
    def check_health(server_url: str = DEFAULT_SERVER_URL) -> Tuple[bool, str]:
        """
        Checks if llama_server is active and returns active model info.
        """
        clean_url = server_url.rstrip("/")
        models_endpoint = f"{clean_url}/v1/models"
        try:
            res = requests.get(models_endpoint, timeout=3)
            if res.status_code == 200:
                data = res.json()
                model_name = "llama_server"
                if "data" in data and len(data["data"]) > 0:
                    model_name = data["data"][0].get("id", model_name)
                elif "models" in data and len(data["models"]) > 0:
                    model_name = data["models"][0].get("name", model_name)
                return True, f"Connected to {model_name}"
            return False, f"Server returned HTTP {res.status_code}"
        except requests.exceptions.RequestException as e:
            return False, f"Connection failed: {str(e)}"

    @staticmethod
    def encode_image(image: Image.Image) -> str:
        """
        Converts PIL Image to base64 JPEG string.
        """
        buffered = io.BytesIO()
        rgb_img = image.convert("RGB")
        rgb_img.save(buffered, format="JPEG", quality=90)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @staticmethod
    def predict(
        server_url: str,
        image: Image.Image,
        prompt: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.1,
        box_thickness: int = 3,
        font_size: int = 14,
        fill_alpha: int = 45,
        timeout: int = 60
    ) -> Tuple[Optional[Image.Image], List[Dict[str, Any]], str, float, Optional[str]]:
        """
        Sends vision grounding request to llama_server and returns annotated image & detections.
        """
        if image is None:
            return None, [], "", 0.0, "Please upload an image."

        if not prompt or not prompt.strip():
            prompt = "locate all objects"

        clean_url = server_url.rstrip("/")
        start_time = time.time()

        # Base64 encode image
        try:
            b64_img = LlamaServerClient.encode_image(image)
        except Exception as e:
            return None, [], "", 0.0, f"Error processing image: {str(e)}"

        # Prepare OpenAI-compatible chat completion payload
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_img}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt.strip()
                        }
                    ]
                }
            ],
            "temperature": temperature
        }

        raw_response_text = ""
        error_msg = None

        try:
            endpoint = f"{clean_url}/v1/chat/completions"
            res = requests.post(endpoint, json=payload, timeout=timeout)
            if res.status_code == 200:
                res_json = res.json()
                raw_response_text = res_json["choices"][0]["message"]["content"]
            else:
                # Fallback to completion endpoint if chat completion endpoint fails
                comp_endpoint = f"{clean_url}/completion"
                comp_payload = {
                    "prompt": f"User: [image: data:image/jpeg;base64,{b64_img}]\n{prompt}\nAssistant:",
                    "temperature": temperature
                }
                res_comp = requests.post(comp_endpoint, json=comp_payload, timeout=timeout)
                if res_comp.status_code == 200:
                    raw_response_text = res_comp.json().get("content", "")
                else:
                    error_msg = f"Server Error {res.status_code}: {res.text}"
        except Exception as e:
            error_msg = f"Network Error: {str(e)}"

        elapsed_time = round(time.time() - start_time, 3)

        if error_msg or not raw_response_text:
            return None, [], raw_response_text or "", elapsed_time, error_msg or "Empty response from server"

        # Parse bounding boxes
        width, height = image.size
        detections = parse_grounding_response(raw_response_text, default_label=prompt)

        # Draw annotations
        annotated_img = draw_annotations(
            image=image,
            detections=detections,
            box_thickness=box_thickness,
            font_size=font_size,
            fill_alpha=fill_alpha
        )

        # Format detection details for UI table
        detection_records = [d.to_dict(width, height) for d in detections]

        return annotated_img, detection_records, raw_response_text, elapsed_time, None
