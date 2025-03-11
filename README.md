# AI Data Analyzer

A Streamlit-based web application that provides AI-powered data analysis and visualization for CSV, JSON, and Excel files.

## Features

- **File Support**: Upload and analyze CSV, JSON, and Excel (xlsx/xls) files
- **Automated Analysis**: 
  - Data cleaning and preparation
  - Descriptive statistics
  - Correlation analysis
  - Distribution visualizations
  - Categorical data analysis
- **Interactive Visualizations**: Dynamic charts and graphs using matplotlib and seaborn
- **Comprehensive Reports**: Downloadable HTML reports with insights and visualizations

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the provided URL
3. Upload your data file
4. Click "Generate Detailed Analysis" to start the analysis

## Dependencies

- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- requests

## Project Structure

```
AI Analyzer/
├── app.py          # Main application file
├── requirements.txt # Project dependencies
└── .gitignore      # Git ignore file
```

## How It Works

1. **Data Upload**: 
   - Supports multiple file formats
   - Automatic format detection and parsing
   - Basic validation and error handling

2. **Analysis Pipeline**:
   - Initial data overview
   - Missing value analysis
   - Statistical analysis
   - Correlation analysis
   - Distribution analysis
   - Categorical data analysis

3. **Visualization**:
   - Correlation heatmaps
   - Distribution plots
   - Bar charts for categorical data
   - Interactive tabs for different visualizations

4. **Reporting**:
   - Comprehensive analysis report
   - Downloadable HTML format
   - Executive summary
   - Key insights
   - Data overview
   - Detailed analysis

## License

MIT License
