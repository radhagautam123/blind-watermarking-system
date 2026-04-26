# Blind Watermarking Project Starter

This starter project includes:
- Classical blind watermarking baseline using DWT + DCT on the Y channel
- Deep-learning watermarking skeleton for PyTorch/Colab
- Attack functions, metrics, preprocessing utilities
- Streamlit demo app
- Colab notebook scaffold for training

## Run locally

1. Create a virtual environment and activate it.
2. Install dependencies:
   pip install -r requirements.txt
3. Run the classical baseline test:
   python test.py
4. Run the Streamlit app:
   streamlit run app/streamlit_app.py

## Suggested workflow
- VS Code: code, testing, baseline, Streamlit app
- Colab: train deep model, save weights to Drive
