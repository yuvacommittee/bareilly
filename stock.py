# ==================================
# MOBILE-FRIENDLY STOCK PREDICTION AI
# ==================================
# Filename: stock_ai_mobile.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow warnings hide

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Lightweight libraries for mobile
try:
    import yfinance as yf
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta
    import json
except ImportError as e:
    print(f"Installing missing libraries...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'yfinance', 'scikit-learn', 'matplotlib', 'pandas', 'numpy'])

# ==================================
# SIMPLIFIED AI MODEL FOR MOBILE
# ==================================

class MobileStockAI:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.accuracy = 0
        
    def fetch_stock_data(self, symbol="RELIANCE.NS", days=365):
        """Fetch stock data for Indian stocks"""
        print(f"📡 Fetching {symbol} data...")
        
        # For Indian stocks, add .NS suffix if not present
        if not symbol.endswith('.NS') and not symbol.startswith('^'):
            symbol = f"{symbol}.NS"
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            df = yf.download(
                symbol,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False
            )
            
            if df.empty:
                print("❌ No data found!")
                return None
            
            print(f"✅ Data fetched: {len(df)} days")
            return df
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def create_features(self, df):
        """Create simple features for mobile"""
        df = df.copy()
        
        # Basic features
        df['Returns'] = df['Close'].pct_change()
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(window=10).std()
        
        # Price momentum
        df['Momentum'] = df['Close'] - df['Close'].shift(5)
        
        # Volume indicator
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Day of week
        df['Day_of_Week'] = df.index.dayofweek
        
        # Target: Next day's closing price
        df['Target'] = df['Close'].shift(-1)
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        return df
    
    def train_model(self, df):
        """Train a simplified model for mobile"""
        print("🧠 Training AI Model...")
        
        # Features
        features = ['Open', 'High', 'Low', 'Volume', 
                   'MA_5', 'MA_10', 'MA_20', 
                   'Returns', 'Volatility', 'Momentum',
                   'Volume_Change', 'Day_of_Week']
        
        X = df[features]
        y = df['Target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest (lightweight)
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate accuracy
        y_pred = self.model.predict(X_test_scaled)
        mae = np.mean(np.abs(y_pred - y_test))
        self.accuracy = 100 - (mae / y_test.mean() * 100)
        
        print(f"✅ Model trained! Accuracy: {self.accuracy:.2f}%")
        
        return X_test, y_test, y_pred
    
    def predict_next_day(self, df):
        """Predict next day's price"""
        if self.model is None:
            print("❌ Model not trained!")
            return None
        
        # Prepare latest data
        features = ['Open', 'High', 'Low', 'Volume', 
                   'MA_5', 'MA_10', 'MA_20', 
                   'Returns', 'Volatility', 'Momentum',
                   'Volume_Change', 'Day_of_Week']
        
        latest_data = df[features].iloc[-1:].copy()
        
        # Update Day_of_Week for tomorrow
        latest_data['Day_of_Week'] = (latest_data['Day_of_Week'] + 1) % 7
        
        # Scale and predict
        latest_scaled = self.scaler.transform(latest_data)
        prediction = self.model.predict(latest_scaled)[0]
        
        return prediction
    
    def get_recommendation(self, current_price, predicted_price):
        """Generate buy/sell/hold recommendation"""
        change_percent = ((predicted_price - current_price) / current_price) * 100
        
        if change_percent > 2:
            return "🟢 STRONG BUY", change_percent
        elif change_percent > 0.5:
            return "🟡 BUY", change_percent
        elif change_percent < -2:
            return "🔴 STRONG SELL", change_percent
        elif change_percent < -0.5:
            return "🟠 SELL", change_percent
        else:
            return "⚪ HOLD", change_percent

# ==================================
# MOBILE WEB INTERFACE
# ==================================

from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

class MobileAIHandler(BaseHTTPRequestHandler):
    ai_model = None
    current_data = None
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = self.get_main_page()
            self.wfile.write(html.encode())
            
        elif self.path == '/data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            if self.current_data is not None:
                data = {
                    'current_price': float(self.current_data['Close'].iloc[-1]),
                    'prediction': float(self.ai_model.predict_next_day(self.current_data)) 
                    if self.ai_model and self.current_data is not None else 0,
                    'accuracy': float(self.ai_model.accuracy) if self.ai_model else 0
                }
                self.wfile.write(json.dumps(data).encode())
                
        elif self.path == '/train':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            symbol = "RELIANCE"
            self.ai_model = MobileStockAI()
            df = self.ai_model.fetch_stock_data(symbol)
            
            if df is not None:
                df = self.ai_model.create_features(df)
                self.current_data = df
                self.ai_model.train_model(df)
                
            self.wfile.write(b"Model trained successfully!")
            
    def get_main_page(self):
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>📱 Mobile Stock AI</title>
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                    font-family: 'Segoe UI', Arial, sans-serif;
                }
                
                body {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 20px;
                    color: white;
                }
                
                .container {
                    max-width: 500px;
                    margin: 0 auto;
                    background: rgba(255, 255, 255, 0.1);
                    backdrop-filter: blur(10px);
                    border-radius: 20px;
                    padding: 25px;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                }
                
                .header {
                    text-align: center;
                    margin-bottom: 30px;
                }
                
                .header h1 {
                    font-size: 28px;
                    margin-bottom: 10px;
                    color: white;
                }
                
                .header p {
                    color: rgba(255, 255, 255, 0.8);
                    font-size: 14px;
                }
                
                .card {
                    background: rgba(255, 255, 255, 0.15);
                    border-radius: 15px;
                    padding: 20px;
                    margin-bottom: 20px;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }
                
                .input-group {
                    margin-bottom: 15px;
                }
                
                label {
                    display: block;
                    margin-bottom: 8px;
                    font-weight: 600;
                    color: white;
                }
                
                input, select {
                    width: 100%;
                    padding: 12px 15px;
                    border-radius: 10px;
                    border: 2px solid rgba(255, 255, 255, 0.3);
                    background: rgba(255, 255, 255, 0.1);
                    color: white;
                    font-size: 16px;
                }
                
                input::placeholder {
                    color: rgba(255, 255, 255, 0.6);
                }
                
                .btn {
                    width: 100%;
                    padding: 14px;
                    border: none;
                    border-radius: 10px;
                    background: linear-gradient(45deg, #4CAF50, #2E7D32);
                    color: white;
                    font-size: 18px;
                    font-weight: bold;
                    cursor: pointer;
                    transition: all 0.3s;
                    margin-top: 10px;
                }
                
                .btn:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
                }
                
                .btn-red {
                    background: linear-gradient(45deg, #f44336, #c62828);
                }
                
                .btn-blue {
                    background: linear-gradient(45deg, #2196F3, #0D47A1);
                }
                
                .result-card {
                    text-align: center;
                    padding: 25px;
                }
                
                .price {
                    font-size: 42px;
                    font-weight: bold;
                    margin: 10px 0;
                }
                
                .prediction {
                    font-size: 36px;
                    color: #4CAF50;
                    margin: 10px 0;
                }
                
                .accuracy {
                    font-size: 18px;
                    color: #FFC107;
                    margin-top: 20px;
                }
                
                .recommendation {
                    font-size: 24px;
                    padding: 10px;
                    border-radius: 10px;
                    margin: 20px 0;
                    background: rgba(255, 255, 255, 0.2);
                }
                
                .loader {
                    border: 5px solid #f3f3f3;
                    border-top: 5px solid #3498db;
                    border-radius: 50%;
                    width: 50px;
                    height: 50px;
                    animation: spin 2s linear infinite;
                    margin: 20px auto;
                }
                
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                
                .footer {
                    text-align: center;
                    margin-top: 30px;
                    font-size: 12px;
                    color: rgba(255, 255, 255, 0.6);
                }
                
                .stock-list {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 10px;
                    margin-top: 15px;
                }
                
                .stock-btn {
                    padding: 10px;
                    background: rgba(255, 255, 255, 0.1);
                    border: 1px solid rgba(255, 255, 255, 0.3);
                    border-radius: 8px;
                    color: white;
                    cursor: pointer;
                    transition: all 0.3s;
                }
                
                .stock-btn:hover {
                    background: rgba(255, 255, 255, 0.2);
                }
                
                @media (max-width: 480px) {
                    .container {
                        padding: 15px;
                    }
                    .price {
                        font-size: 32px;
                    }
                    .prediction {
                        font-size: 28px;
                    }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📱 Mobile Stock AI</h1>
                    <p>Indian Stock Market Prediction - Educational Purpose</p>
                </div>
                
                <div class="card">
                    <div class="input-group">
                        <label>Select Indian Stock:</label>
                        <div class="stock-list">
                            <button class="stock-btn" onclick="selectStock('RELIANCE')">RELIANCE</button>
                            <button class="stock-btn" onclick="selectStock('TCS')">TCS</button>
                            <button class="stock-btn" onclick="selectStock('HDFCBANK')">HDFC</button>
                            <button class="stock-btn" onclick="selectStock('INFY')">INFY</button>
                            <button class="stock-btn" onclick="selectStock('ITC')">ITC</button>
                            <button class="stock-btn" onclick="selectStock('SBIN')">SBI</button>
                        </div>
                        <input type="text" id="stockInput" placeholder="Enter stock symbol (e.g., RELIANCE)" value="RELIANCE">
                    </div>
                    
                    <button class="btn btn-blue" onclick="trainModel()">🚀 Train AI Model</button>
                    <button class="btn" onclick="getPrediction()">🔮 Get Prediction</button>
                    <button class="btn btn-red" onclick="location.reload()">🔄 Reset</button>
                </div>
                
                <div class="card result-card">
                    <h3>Current Analysis</h3>
                    <div id="loader" class="loader" style="display:none;"></div>
                    
                    <div id="results">
                        <div class="price">₹ 0.00</div>
                        <small>Current Price</small>
                        
                        <div class="prediction">₹ 0.00</div>
                        <small>Predicted Price</small>
                        
                        <div class="recommendation" id="recommendation">
                            ⚪ NO DATA
                        </div>
                        
                        <div class="accuracy" id="accuracy">
                            Accuracy: 0%
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h3>⚠️ Important Disclaimer</h3>
                    <p style="font-size: 12px; line-height: 1.5;">
                        This is for EDUCATIONAL purposes only. Stock predictions are never 100% accurate. 
                        Past performance doesn't guarantee future results. You are solely responsible 
                        for your investment decisions.
                    </p>
                </div>
                
                <div class="footer">
                    <p>Mobile Stock AI v1.0 | Made for Educational Purpose</p>
                    <p>Server running on: <span id="serverUrl">localhost:8080</span></p>
                </div>
            </div>
            
            <script>
                function selectStock(symbol) {
                    document.getElementById('stockInput').value = symbol;
                }
                
                function trainModel() {
                    document.getElementById('loader').style.display = 'block';
                    document.getElementById('results').style.display = 'none';
                    
                    fetch('/train')
                        .then(response => response.text())
                        .then(data => {
                            alert('Model trained successfully!');
                            document.getElementById('loader').style.display = 'none';
                            document.getElementById('results').style.display = 'block';
                            getPrediction();
                        })
                        .catch(error => {
                            alert('Error training model: ' + error);
                            document.getElementById('loader').style.display = 'none';
                        });
                }
                
                function getPrediction() {
                    fetch('/data')
                        .then(response => response.json())
                        .then(data => {
                            const currentPrice = data.current_price || 0;
                            const predictedPrice = data.prediction || 0;
                            const accuracy = data.accuracy || 0;
                            
                            // Update UI
                            document.querySelector('.price').textContent = '₹ ' + currentPrice.toFixed(2);
                            document.querySelector('.prediction').textContent = '₹ ' + predictedPrice.toFixed(2);
                            document.getElementById('accuracy').textContent = 
                                'Accuracy: ' + accuracy.toFixed(2) + '%';
                            
                            // Generate recommendation
                            const change = ((predictedPrice - currentPrice) / currentPrice) * 100;
                            let recommendation = '';
                            
                            if (change > 2) recommendation = '🟢 STRONG BUY (+' + change.toFixed(2) + '%)';
                            else if (change > 0.5) recommendation = '🟡 BUY (+' + change.toFixed(2) + '%)';
                            else if (change < -2) recommendation = '🔴 STRONG SELL (' + change.toFixed(2) + '%)';
                            else if (change < -0.5) recommendation = '🟠 SELL (' + change.toFixed(2) + '%)';
                            else recommendation = '⚪ HOLD (' + change.toFixed(2) + '%)';
                            
                            document.getElementById('recommendation').textContent = recommendation;
                        })
                        .catch(error => {
                            console.error('Error:', error);
                            alert('Please train the model first!');
                        });
                }
                
                // Get current IP for display
                window.onload = function() {
                    fetch('/data').catch(() => {});
                };
            </script>
        </body>
        </html>
        """

# ==================================
# START MOBILE SERVER
# ==================================

def start_mobile_server(port=8080):
    """Start HTTP server for mobile interface"""
    print(f"\n{'='*50}")
    print("📱 MOBILE STOCK PREDICTION AI STARTING...")
    print('='*50)
    
    # Create model instance
    ai = MobileStockAI()
    MobileAIHandler.ai_model = ai
    
    # Try to get local IP
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "127.0.0.1"
    
    server = HTTPServer(('0.0.0.0', port), MobileAIHandler)
    
    print(f"\n✅ Server started successfully!")
    print(f"\n📱 Open in Mobile Browser:")
    print(f"   http://{local_ip}:{port}")
    print(f"   OR")
    print(f"   http://127.0.0.1:{port}")
    print(f"\n🌐 Open in PC Browser (same network):")
    print(f"   http://{local_ip}:{port}")
    print(f"\n🔄 Server running... Press Ctrl+C to stop")
    print('='*50)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n❌ Server stopped by user")
        server.server_close()

# ==================================
# COMMAND LINE INTERFACE
# ==================================

def run_cli_mode():
    """Run AI in command line mode"""
    print("\n📊 STOCK PREDICTION AI - CLI MODE")
    print("="*40)
    
    stocks = {
        '1': 'RELIANCE',
        '2': 'TCS', 
        '3': 'HDFCBANK',
        '4': 'INFY',
        '5': 'ITC',
        '6': 'SBIN',
        '7': 'ICICIBANK',
        '8': 'BHARTIARTL'
    }
    
    print("\nAvailable Indian Stocks:")
    for key, value in stocks.items():
        print(f"  {key}. {value}")
    
    choice = input("\nSelect stock (1-8): ").strip()
    
    if choice in stocks:
        symbol = stocks[choice]
        print(f"\nAnalyzing {symbol}...")
        
        # Initialize and run AI
        ai = MobileStockAI()
        
        # Fetch data
        df = ai.fetch_stock_data(symbol)
        if df is None:
            print("❌ Failed to fetch data!")
            return
        
        # Create features
        df = ai.create_features(df)
        
        # Train model
        X_test, y_test, y_pred = ai.train_model(df)
        
        # Make prediction
        current_price = df['Close'].iloc[-1]
        predicted_price = ai.predict_next_day(df)
        
        if predicted_price:
            recommendation, change = ai.get_recommendation(current_price, predicted_price)
            
            print(f"\n{'='*40}")
            print("📈 PREDICTION RESULTS:")
            print(f"{'='*40}")
            print(f"Stock: {symbol}")
            print(f"Current Price: ₹{current_price:.2f}")
            print(f"Predicted Next Day: ₹{predicted_price:.2f}")
            print(f"Expected Change: {change:.2f}%")
            print(f"\nRecommendation: {recommendation}")
            print(f"Model Accuracy: {ai.accuracy:.2f}%")
            print(f"{'='*40}")
            
            # Plot simple graph
            try:
                plt.figure(figsize=(10, 6))
                plt.plot(y_test.values[-30:], label='Actual', marker='o')
                plt.plot(y_pred[-30:], label='Predicted', marker='x')
                plt.title(f'{symbol} - Actual vs Predicted Prices')
                plt.xlabel('Days')
                plt.ylabel('Price (₹)')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                
                # Save and show plot
                plt.savefig('/sdcard/stock_prediction.png')
                print("\n📊 Graph saved as: /sdcard/stock_prediction.png")
                plt.show()
            except:
                print("\n📊 Graph display not available in this environment")
    else:
        print("❌ Invalid choice!")

# ==================================
# MAIN EXECUTION
# ==================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🤖 MOBILE STOCK PREDICTION AI")
    print("="*50)
    print("\nSelect Mode:")
    print("  1. Web Interface (Browser)")
    print("  2. Command Line Interface")
    print("  3. Quick Test (RELIANCE)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        port = input("Enter port number (default 8080): ").strip()
        port = int(port) if port.isdigit() else 8080
        start_mobile_server(port)
        
    elif choice == '2':
        run_cli_mode()
        
    elif choice == '3':
        print("\nRunning quick test on RELIANCE...")
        ai = MobileStockAI()
        df = ai.fetch_stock_data("RELIANCE")
        if df is not None:
            df = ai.create_features(df)
            ai.train_model(df)
            pred = ai.predict_next_day(df)
            print(f"\n✅ Quick Test Complete!")
            print(f"Current Price: ₹{df['Close'].iloc[-1]:.2f}")
            print(f"Predicted Next Day: ₹{pred:.2f}")
        else:
            print("❌ Test failed!")
            
    else:
        print("❌ Invalid choice!")
