# run_ai_pipeline.py
import os
import pandas as pd
from ai_engine.deep_predictor import DeepPredictorEngine


def main():
  print("Initializing AI Deep Learning Engine...")

  # 1. Start up the AI brain (it automatically loads 'ai_engine/model_checkpoint.pt' if it exists)
  ai_brain = DeepPredictorEngine(model_path="ai_engine/model_checkpoint.pt")

  calculated_dir = "calculated_output"

  if not os.path.exists(calculated_dir):
    print(
        f"Error: '{calculated_dir}' folder not found. Please run your indicator"
        " calculation script first!"
    )
    return

  # 2. Get all calculated stock files
  stock_files = [f for f in os.listdir(calculated_dir) if f.endswith(".csv")]

  if not stock_files:
    print(f"No stock CSV files found in '{calculated_dir}'.")
    return

  print(
      f"Found {len(stock_files)} stocks ready for AI analysis and continual"
      " learning.\n"
  )

  # 3. Loop through every stock one by one
  for index, filename in enumerate(stock_files, 1):
    ticker = filename.replace(".csv", "")
    file_path = os.path.join(calculated_dir, filename)

    print(f"[{index}/{len(stock_files)] Processing AI for: {ticker}")

    try:
      # Load the stock's calculated indicators and price data
      df = pd.read_csv(file_path, parse_dates=["Datetime"])

      # Ensure we have enough rows for the AI sequence window (at least 35 bars)
      if len(df) < 35:
        print(f"  -> Skipping {ticker}: Not enough data rows yet.")
        continue

      # A. PREDICTION STEP: Ask the AI what it thinks the next minute will look like
      next_bar = ai_brain.predict_next_bar(df, seq_length=30)
      if next_bar:
        print(f"  -> [AI Forecast for Next Bar]:")
        print(
            f"     Predicted Close: {next_bar['Predicted_Close']:.2f} |"
            f" Predicted Volume: {next_bar['Predicted_Volume']:.0f}"
        )

      # B. LEARNING STEP: Feed historical data so the AI compares its past guesses to reality
      # This updates the neural network weights via backpropagation and saves them
      ai_brain.learn_from_reality(df, epochs=2, seq_length=30)

    except Exception as e:
      print(f"  -> Error processing AI for {ticker}: {e}")

  print(
      "\nAll stock files processed! The AI brain has updated its weights and"
      " saved them to 'ai_engine/model_checkpoint.pt'."
  )


if __name__ == "__main__":
  main()