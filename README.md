
# LLM Test Chatbot

This project demonstrates a LLM chatbot using Hugging Face models and Gradio. It allows users to dynamically switch between different HF format models stored in a specified directory and interact with the chatbot through a web interface.

## Features

- Load and use different Hugging Face models stored locally.
- Dynamic model switching without restarting the application.
- Interactive web interface using Gradio.
- Efficient memory and GPU usage by releasing resources when switching models.

## Prerequisites

- Python 3.11 or higher
- PyTorch with GPU support (if available)
- Required Python packages: `torch`, `transformers`, `gradio`

## Installation

1. Clone the repository:

```bash
git clone https://github.com/SmallBlueE/LLMChatbot.git
cd LLMChatbot
```

2. Install the required packages:

```bash
pip install torch torchvision torchaudio transformers gradio
```

3. Ensure you have the Hugging Face format models stored in a directory (e.g., MODELS_DIR=`d:/models`). The directory should contain subdirectories for each model.

## Usage

1. Run the chatbot application:

```bash
python chatbot.py
```

2. Open the provided URL in your web browser to interact with the chatbot.

## Code Explanation

The main script `chatbot.py` contains the implementation of the dynamic model switching chatbot. Here are some key parts of the code:


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.

---

By following these instructions, users will be able to set up and run the LLM chatbot on their local machines.
