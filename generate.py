import streamlit as st
import os
import json
import time
import requests
from dotenv import load_dotenv

def generate_image(
    finetune_id,
    api_key,
    prompt="A majestic image of a TOK",
    finetune_strength=1.1,
    steps=40,
    guidance=2.5,
    width=512,
    height=512,
    seed=None,
    safety_tolerance=2,
    output_format="jpeg"
):
    """
    Submit an inference request for an image generation task using the specified finetune.
    Returns a JSON response with an 'id' field for the inference job, e.g.:
    {
      "id": "53dca870-eadc-446f-aa5d-aec649c12e26",
      ...
    }
    """
    url = "https://api.us1.bfl.ai/v1/flux-pro-finetuned"
    headers = {
        "Content-Type": "application/json",
        "X-Key": api_key,
    }
    payload = {
        "finetune_id": finetune_id,
        "finetune_strength": finetune_strength,
        "prompt": prompt,
        "steps": steps,
        "guidance": guidance,
        "width": width,
        "height": height,
        "safety_tolerance": safety_tolerance,
        "output_format": output_format,
    }
    # Only include seed if user provided one
    if seed is not None:
        payload["seed"] = seed

    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()

def check_inference(inference_id, api_key):
    """
    Poll for inference status until it's "Ready" or we hit
    another terminal status (e.g. Error, Moderation), then return the final response.
    Shows a progress bar when progress is between 0 and 1.
    """
    url = "https://api.us1.bfl.ai/v1/get_result"
    headers = {
        "Content-Type": "application/json",
        "X-Key": api_key,
    }
    params = {"id": inference_id}

    # Initialize Streamlit progress bar
    progress_bar = st.progress(0)

    while True:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()

        status = data.get("status")
        progress_value = data.get("progress")

        if status == "Ready":
            st.success("Inference complete!")
            return data

        elif status == "Task not found":
            st.error("Task not found! Check if the ID is correct or if the job has expired.")
            return data

        elif status == "Request Moderated":
            st.error("The request was flagged by moderation. Generation stopped.")
            return data

        elif status == "Content Moderated":
            st.error("The generated content was flagged by moderation. Generation stopped.")
            return data

        elif status == "Error":
            st.error("An error occurred while generating the image. Check the logs or try again.")
            return data

        else:
            # Show progress if it's a float between 0 and 1
            if isinstance(progress_value, (int, float)) and 0 <= progress_value <= 1:
                progress_bar.progress(progress_value)
                st.info(f"Inference in progress: {progress_value * 100:.0f}%")
            else:
                # If no numeric progress is available, just show the status
                st.info(f"Current status: {status}. Checking again in 3s...")
            time.sleep(3)

def read_finetunes():
    """
    Load the stored fine-tune IDs from finetune_id.json (if it exists),
    and return them as a dict: {finetune_comment: finetune_id}
    """
    file_path = "finetune_id.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return {}  # No file found, or it's empty

def main():
    load_dotenv()
    default_api_key = os.getenv("BFL_API_KEY", "")

    st.title("BFL Generate Images with Finetunes")
    st.write("Pick a fine-tuned model and generate new images in style, product, or general modes.")

    api_key = st.text_input("Enter your BFL API key", value=default_api_key, type="password")
    if not api_key:
        st.warning("API key is required.")
        st.stop()

    # Load any existing finetunes
    finetunes = read_finetunes()
    if not finetunes:
        st.warning("No fine-tunes found. Please run the finetune app first.")
        st.stop()

    # Let user pick from available fine-tunes
    finetune_comment = st.selectbox("Choose a Fine-Tune", list(finetunes.keys()))
    finetune_id = finetunes[finetune_comment]

    # Basic image generation parameters
    prompt = st.text_input("Prompt", value="image of a TOK")
    finetune_strength = st.slider("Finetune Strength", min_value=0.0, max_value=2.0, value=1.1, step=0.1)
    steps = st.slider("Steps", min_value=1, max_value=50, value=40)
    guidance = st.slider("Guidance Scale", min_value=1.5, max_value=5.0, value=2.5, step=0.1)
    width = st.selectbox("Width", [256, 512, 768, 1024, 1280, 1344, 1440], index=1)
    height = st.selectbox("Height", [256, 512, 768, 1024, 1280, 1344, 1440], index=2)
    seed = st.text_input("Seed (optional; leave blank for random)")

    # Safety tolerance & output format
    safety_tolerance = st.slider("Safety Tolerance", min_value=0, max_value=6, value=2)
    output_format = st.selectbox("Output Format", ["jpeg", "png"])

    if st.button("Generate Image"):
        # Convert seed to int if user provided one
        if seed.strip() == "":
            seed_int = None
        else:
            try:
                seed_int = int(seed)
            except ValueError:
                st.error("Seed must be an integer.")
                return

        st.write("Submitting generation request...")
        try:
            # Fire off an image generation request
            generate_response = generate_image(
                finetune_id=finetune_id,
                api_key=api_key,
                prompt=prompt,
                finetune_strength=finetune_strength,
                steps=steps,
                guidance=guidance,
                width=width,
                height=height,
                seed=seed_int,
                safety_tolerance=safety_tolerance,
                output_format=output_format,
            )
            # This response should contain the inference task id
            inference_id = generate_response.get("id")
            if not inference_id:
                st.error("No inference_id returned from the server. Check logs or try again.")
                return

            st.success(f"Generation request submitted! Inference ID: {inference_id}")
            st.write("Polling for result...")

            # Poll until "Ready" or until we hit a terminal status
            final_data = check_inference(inference_id, api_key=api_key)
            result = final_data.get("result", {})

            # The 'result' is often JSON in string form, so parse it if needed
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    st.error(f"Unexpected result format: {result}")
                    return

            sample_url = result.get("sample")
            if sample_url:
                st.image(sample_url, caption="Your Generated Image", use_container_width=True)
            else:
                st.error("No image URL found in the result.")

        except Exception as e:
            st.error(f"Error generating image: {str(e)}")

if __name__ == "__main__":
    main()