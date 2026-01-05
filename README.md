# Financial Analyst Copilot

A comprehensive SaaS Financial Analysis tool that combines traditional metrics visualization with AI-powered insights. This application helps financial analysts and business owners track key revenue metrics and generate automated explanations using Large Language Models (LLMs).

## 🚀 Features

- **Interactive Dashboard**: Built with Streamlit for real-time interaction.
- **Key Metrics Visualization**: Track important SaaS metrics including:
  - Total MRR (Monthly Recurring Revenue)
  - New MRR, Expansion MRR, Contraction MRR, Churn MRR
  - Net New MRR
  - Active Customers
  - Revenue Churn Rate
- **Time Window Filtering**: Analyze data over different periods (Last 6 months, 12 months, 24 months, or All time).
- **Automated Insights**:
  - Month-over-Month (MoM) calculations.
  - Identification of Best and Worst performing months.
- **AI Copilot**: Integrated LLM (Ollama or OpenRouter) to explain trends, answer specific questions about the data, and provide qualitative analysis.

## 🛠️ Tech Stack

- **Python**: Core programming language.
- **Streamlit**: utilized for the web interface and dashboard.
- **Pandas**: For data manipulation and analysis.
- **LLM Integration**:
  - **Ollama**: For local LLM inference.
  - **OpenRouter**: For accessing various cloud-based models.

## 📋 Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) (optional, for local LLM support)

## 📦 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/financial-analyst-copilot.git
   cd financial-analyst-copilot
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configuration

Create a `.env` file in the root directory of the project to configure your LLM provider.

### Option 1: Using Ollama (Local)

Ensure Ollama is running (`ollama serve`).

```ini
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=phi3.5:3.8b-mini-instruct-q4_K_M
```

### Option 2: Using OpenRouter (Cloud)

```ini
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=openai/gpt-3.5-turbo
# Optional
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

## 🚀 Usage

1. **Ensure your `.env` file is configured.**
2. **Make sure your data is in the `data/` directory.** (The application expects CSV files for customer MRR and revenue events).
3. **Run the Streamlit application:**

   ```bash
   streamlit run src/ui/app.py
   ```

4. **Access the dashboard** in your browser at `http://localhost:8501`.

## 📂 Project Structure

```
financial-analyst-copilot/
├── data/               # Data files (CSVs)
├── notebooks/          # Exploratory Jupyter notebooks
├── src/
│   ├── llm/            # LLM client and factory logic
│   ├── metrics/        # Core metrics calculation logic
│   ├── ui/             # Streamlit application
│   └── ...
├── .env                # Configuration file
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

[MIT](LICENSE)
