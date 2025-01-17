# BFL FLUX Pro Fine-Tuning & Image Generation

This project provides two Streamlit apps:

1. **`finetune.py`**  
   Helps you fine-tune a model using Black Forest Labs’ new FLUX Pro finetune capabilities.
2. **`generate.py`**  
   Lets you generate images from your newly fine-tuned models.  

Feel free to use or expand on this as you see fit!

---

## Features

- **Easy Fine-Tuning**: Upload a `.zip` of your training data, configure advanced settings (iterations, learning rate, etc.), and submit.
- **Progress Polling**: The app checks your finetune job until it’s ready, then stores the ID in a `finetune_id.json`.
- **Image Generation**: Select a finetune you created, provide a prompt, and generate images.
- **Progress Bar**: Watch the progress of your image generation as it’s happening.

---

## Setup Instructions

### 0. Refer to the Official FLUX Documentation

You can refer to this documentation for guidance on the various options/parameters, as well as how to create the .zip file of training data:
https://docs.bfl.ml/finetuning/

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/bfl-flux-finetune.git
cd bfl-flux-finetune
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# or
.\.venv\Scripts\activate   # On Windows
```

### 3. Install Dependencies

I recommend installing via pip, and installing via the requirements.txt:

```bash
pip install -r requirements.txt
```
That should handle the primary libraries this project uses.

## Configuring Your API Key

You’ll need a .env file in the root of this project with your Black Forest Labs API key. Create a file named .env containing:

```
BFL_API_KEY=YOUR_BFL_API_KEY
```

Replace YOUR_BFL_API_KEY with your actual API key from Black Forest Labs.

## Running the Finetune App

To launch the finetune interface:

```bash
streamlit run finetune.py
```

You’ll be prompted for:

1. API Key: If not provided via .env, you can enter it in the UI.
2. ZIP File: A .zip containing your training images.
3. Fine-tune Options: Comment/Name, Trigger Word, Mode (character, product, style, general), etc.
4. Advanced Settings: Iterations, learning rate, captioning, priority, etc.

The app creates a temp folder if it doesn’t already exist and stores your ZIP data there. Then it submits the request to Black Forest Labs. After submission, it will poll for status. Once finished, you’ll see “Finetuning is complete!” The resulting fine-tune ID is stored in finetune_id.json.

## Running the Image Generation App

To launch the image generation interface:

```bash
streamlit run generate.py
```

You’ll be prompted for:

1. API Key: Again, if not provided via .env, you can enter it in the UI.
2. Fine-tune Selection: Pick one from the IDs in finetune_id.json.
3. Prompt Settings: Prompt, finetune strength, steps, guidance, size, and optional seed.
4. Execution: Click “Generate Image” and the app will poll your job, showing a progress bar. Once complete, the final image is displayed.

## Notes & Tips
- Overwriting Temp Files: If temp_finetune_data.zip already exists in temp, it’ll be overwritten each time you submit new training data.
- Modifying Defaults: The default iteration count is 300, default learning rate is chosen by the server (1e-5 for full, 1e-4 for lora), and default rank is 32. Adjust in the UI or code if needed.
- Storage: After every finetune, the finetune ID is stored in finetune_id.json using the “comment” or name you specified. That’s how the generation app knows what fine-tunes are available.

## License

This project is licensed under the MIT License. Feel free to share, modify, and reuse under the terms specified therein.

Enjoy fine-tuning with Black Forest Labs FLUX Pro!
