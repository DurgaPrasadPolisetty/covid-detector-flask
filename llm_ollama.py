import os
import requests
import json
import logging
import time

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_explanation(prediction, confidence=None):
    """
    Generates a detailed medical explanation for the detected condition focusing on X-ray findings.
    """
    confidence_str = f" with {confidence:.2f}% confidence" if confidence else ""
    
    prompt = f"""
    Context: You are an expert radiologist providing patient-friendly explanations of chest X-ray findings.
    
    Patient Input: The user has uploaded a chest X-ray scan image. This is a standard radiographic image of the patient's chest showing the lungs, heart, and related structures.
    
    Finding Detected: {prediction}{confidence_str}
    
    Task: Explain what this finding means by describing:
    1. What visible changes or patterns in the X-ray led to this finding
    2. What the patient should understand about these X-ray features
    3. Why this finding is important and next steps
    
    Constraint: Keep the explanation professional, clear, and easy to understand for patients (max 4 sentences). Focus on X-ray features and what they indicate, not technical model details.
    
    Important: Always include a clear disclaimer that this is an AI-generated analysis of the X-ray and should be verified by a qualified healthcare professional.
    """
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3
        }
    }

    try:
        url = f"{OLLAMA_BASE_URL}/api/generate"
        logger.info(f"Requesting LLM explanation for: {prediction}")
        logger.info(f"Connecting to Ollama at: {url}")
        logger.info(f"OLLAMA_BASE_URL value: {OLLAMA_BASE_URL}")
        logger.info(f"Payload: {payload}")
        
        r = requests.post(
            url,
            json=payload,
            timeout=(10, 300)
        )
        logger.info(f"Response status: {r.status_code}")
        r.raise_for_status()
        data = r.json()
        logger.info(f"Got response from Ollama: {data.get('response', '')[:100]}")
        return data.get("response", "No explanation available.")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"LLM Connection Error: {e}", exc_info=True)
        return f"Analysis: {prediction} detected. (Unable to connect to AI Assistant. Please try again or consult a healthcare professional.)"
    except requests.exceptions.Timeout as e:
        logger.error(f"LLM Timeout: {e}", exc_info=True)
        return f"Analysis: {prediction} detected. (AI Assistant response timeout. Please try again.)"
    except Exception as e:
        logger.error(f"LLM Error: {e}", exc_info=True)
        return f"Analysis: {prediction} detected. (AI Assistant currently offline. Please consult a healthcare professional.)"

def generate_diagnostic_reasoning(prediction, metadata=None):
    """
    Generates a more technical reasoning for the diagnosis (e.g., for doctors).
    """
    prompt = f"""
    Context: Clinical Decision Support System.
    Subject: Chest X-ray finding: {prediction}.
    Task: Provide a technical summary of typical radiographic features associated with this finding.
    Tone: Medical professional.
    """
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=(10, 300)
        )
        r.raise_for_status()
        return r.json().get("response", "")
    except:
        return "Technical reasoning unavailable."
