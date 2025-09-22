# Stock Price Prediction w/ LSTM

This project builds a time-series prediction model in PyTorch to forecast stock prices using LSTMs (Long Short-Term Memory networks). It demonstrates data preprocessing, model training, and evaluation within Jupyter Lab.

**Workflow**
1. Data Collection
   - Pulls stock data (ex. Apple = ```APPL```) from Yahoo Finance via ```yfinance```

2. Preprocessing
   - Standardizes values with ```StandardScaler``` to normalize data
   - Transforms the continuous series into overlapping 30-day sliding windows to give the LSTM temporal context

3. Train/Test Split
   - 80% of sequences for training, 20% for testing
   - Inputs (```X```) are prior 29 days, outputs (```y```) are the 30th day's closing price

4. Model Architecture
   - 2-layer LSTM with hidden size 32
   - Fully connected layer maps hidden state -> next day's price
   - Trained with MSE loss and Adam optimizer

5. Training
   - Runs for 200 epochs
   - Backpropogation with gradient updates ever pass
   - Prints loss every 25 epochs for tracking convergence
  
6. Evaluation
   - Preductions rescaled to original price units
   - Computes RMSE for both training and test sets
   - Visualizes:
        - Predicted vs Actual stock prices
        - Prediction error over time

**Results**
- Produces next-day price predictions from 30-day input windows
- Shows how well LSTMs capture temporal patterns in stock data.

**Tech Stack**
- Python (Jupyter Lab)
- Libraries: NumPy, PyTorch, Pandas, Matplotlib, scikit-learn, yfinance
