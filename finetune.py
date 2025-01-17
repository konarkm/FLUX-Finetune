import streamlit as st
import os
import time
import json
import base64
import requests
from dotenv import load_dotenv

def request_finetuning(
    zip_path,
    finetune_comment,
    trigger_word="TOK",
    mode="character",
    api_key=None,
    iterations=300,
    learning_rate=None,
    captioning=True,
    priority="quality",
    finetune_type="full",
    lora_rank=32,
):
    """
    Request a finetuning using the provided ZIP file.

    Args:
        zip_path (str): Path to the ZIP file containing training data
        finetune_comment (str): Comment for the finetune_details
        trigger_word (str): Trigger word for the model
        mode (str): Mode for caption generation [character, product, style, general]
        api_key (str): API key for authentication
        iterations (int): Number of training iterations
        learning_rate (float or None): Learning rate (None means let the server pick defaults)
        captioning (bool): Enable/disable auto-captioning
        priority (str): Training priority ('speed' or 'quality')
        finetune_type (str): "full" or "lora"
        lora_rank (int): Lora rank (16 or 32)

    Returns:
        dict: API response
    """
    if api_key is None:
        if "BFL_API_KEY" not in os.environ:
            raise ValueError(
                "Provide your API key via --api_key or an environment variable BFL_API_KEY"
            )
        api_key = os.environ["BFL_API_KEY"]

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file not found at {zip_path}")

    assert mode in ["character", "product", "style", "general"], \
        "Mode must be one of [character, product, style, general]."

    # Read & base64 encode our zip
    with open(zip_path, "rb") as file:
        encoded_zip = base64.b64encode(file.read()).decode("utf-8")

    url = "https://api.us1.bfl.ai/v1/finetune"
    headers = {
        "Content-Type": "application/json",
        "X-Key": api_key,
    }
    payload = {
        "finetune_comment": finetune_comment,
        "trigger_word": trigger_word,
        "file_data": encoded_zip,
        "iterations": iterations,
        "mode": mode,
        "captioning": captioning,
        "priority": priority,
        "lora_rank": lora_rank,
        "finetune_type": finetune_type,
    }

    # Only include learning_rate if user provided one
    if learning_rate is not None:
        payload["learning_rate"] = learning_rate

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

def finetune_progress(
    finetune_id,
    api_key=None,
):
    """
    Query the status of a given finetune_id. Returns JSON that includes:
    {
      "id": <finetune_id>,
      "status": "Pending" | "Ready" | ...
      "result": ...,
      "progress": ...
    }
    """
    if api_key is None:
        if "BFL_API_KEY" not in os.environ:
            raise ValueError(
                "Provide your API key via --api_key or an environment variable BFL_API_KEY"
            )
        api_key = os.environ["BFL_API_KEY"]

    url = "https://api.us1.bfl.ai/v1/get_result"
    headers = {
        "Content-Type": "application/json",
        "X-Key": api_key,
    }
    payload = {
        "id": finetune_id,
    }

    response = requests.get(url, headers=headers, params=payload)
    response.raise_for_status()
    return response.json()

def store_finetune_id(finetune_comment, finetune_id):
    """
    Store the fine-tune ID in a JSON file, keyed by finetune_comment.
    If the file doesn't exist, create it. If it does, append to it.
    """
    file_path = "finetune_id.json"
    
    # Load existing data (if any)
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    # Update our local dictionary
    data[finetune_comment] = finetune_id

    # Write back to disk
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)

def main():
    # Load BFL_API_KEY from .env if available
    load_dotenv()
    default_api_key = os.getenv("BFL_API_KEY", "")

    st.title("BFL Finetune App")
    st.write("Fine-tuning is fun... until it isn't. Let's do it anyway.")

    # Let the user override .env's key if they want
    api_key = st.text_input("Enter your BFL API key", value=default_api_key, type="password")
    if not api_key:
        st.warning("API key is required. Gotta keep those credentials safe.")
        st.stop()

    # Let the user upload their zip file
    uploaded_file = st.file_uploader("Pick your finetuning zip file here", type=["zip"])
    
    # Basic required inputs
    finetune_comment = st.text_input("Finetune comment", value="My first finetune")
    trigger_word = st.text_input("Trigger word", value="TOK")
    mode = st.selectbox("Finetune mode", ["character", "product", "style", "general"])

    # Advanced options (expandable)
    with st.expander("Advanced Fine-Tune Options"):
        iterations = st.number_input(
            "Iterations",
            min_value=100,
            max_value=1000,
            value=300,
            step=10
        )
        # A helper to allow "None" as a valid "no input" for learning_rate
        use_custom_lr = st.checkbox("Specify a custom learning rate?")
        lr_value = None
        if use_custom_lr:
            lr_value = st.number_input(
                "Learning Rate (leave unchecked to use server defaults)",
                min_value=1e-7,
                max_value=0.5,
                value=1e-5,
                format="%.7f"
            )
        captioning = st.checkbox("Enable Captioning?", value=True)
        priority = st.selectbox("Priority", ["speed", "quality"], index=1)
        finetune_type = st.selectbox("Finetune Type", ["full", "lora"], index=0)
        lora_rank = st.selectbox("Lora Rank", [16, 32], index=1)

    # If everything is filled out, do the big show
    if st.button("Submit for Finetuning"):
        if not uploaded_file:
            st.warning("No file, no finetune. Upload a .zip first.")
            st.stop()
        
        # Create "temp" directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)

        # Save the uploaded file in the "temp" folder
        temp_zip_path = os.path.join("temp", "finetune_data.zip")
        with open(temp_zip_path, "wb") as temp_zip:
            temp_zip.write(uploaded_file.getbuffer())

        try:
            # Request finetuning
            response = request_finetuning(
                zip_path=temp_zip_path,
                finetune_comment=finetune_comment,
                trigger_word=trigger_word,
                mode=mode,
                api_key=api_key,
                iterations=iterations,
                learning_rate=lr_value,
                captioning=captioning,
                priority=priority,
                finetune_type=finetune_type,
                lora_rank=lora_rank,
            )
            st.success("Finetuning request submitted successfully!")
            st.json(response)

            finetune_id = response.get("finetune_id")
            if not finetune_id:
                st.warning("No finetune ID found in response. Can't check progress.")
                return

            # Store the finetune ID to a JSON file
            store_finetune_id(finetune_comment, finetune_id)

            st.write("Sit tight! We'll keep poking the server every 10s to see if it's ready.")

            # We poll until status is "Ready"
            while True:
                progress = finetune_progress(finetune_id, api_key=api_key)
                status = progress.get("status")
                if status == "Ready":
                    st.success("Finetuning is complete! Go forth and conquer!")
                    break
                else:
                    st.info(f"Current status: {status}. Waiting 10s...")
                    time.sleep(10)

        except Exception as e:
            st.error(f"Error requesting finetune: {str(e)}")

if __name__ == "__main__":
    main()