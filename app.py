import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import json
import requests
import base64
from io import StringIO
import re

st.set_page_config(page_title="AI Data Analyzer", layout="wide")

st.title("🧠 AI Data Analyzer")
st.markdown("Upload your data file and get an AI-powered comprehensive analysis report with visualizations.")

# File upload
uploaded_file = st.file_uploader("Choose a file", type=["csv", "json", "xlsx", "xls"])

@st.cache_data
def get_initial_data_info(df):
    """Extract initial information about the dataset without loading entire data into memory"""
    info = {
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "shape": df.shape,
        "sample_data": df.head(5).to_dict(),
        "numerical_columns": df.select_dtypes(include=['int64', 'float64']).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object', 'bool', 'category']).columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "summary_stats": df.describe().to_dict()
    }
    return info

def get_analysis_plan(data_info):
    """Get analysis plan from LLM based on data information"""
    prompt = f"""
    I have a dataset with the following characteristics:
    - Shape: {data_info['shape']}
    - Columns: {data_info['columns']}
    - Data types: {data_info['dtypes']}
    - Numerical columns: {data_info['numerical_columns']}
    - Categorical columns: {data_info['categorical_columns']}
    - Missing values: {data_info['missing_values']}
    
    Generate a comprehensive analysis plan with specific Python code blocks. Each code block should be complete and executable, focusing on:
    1. Data cleaning and preparation
    2. Descriptive statistics
    3. Visualizations (using matplotlib and seaborn)
    4. Feature relationships and correlations
    5. Pattern identification
    
    Format your response as a JSON with keys being the analysis step name and values being the executable Python code.
    """
    
    try:
        # Use proper URL encoding for the prompt
        encoded_prompt = requests.utils.quote(prompt)
        # Log for debugging
        st.write(f"Sending request to API with prompt length: {len(prompt)}")
        
        # Make the API request with proper headers and in the request body instead
        response = requests.post(
            "https://text.pollinations.ai/api/generate", 
            json={"prompt": prompt},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        # Log the response status and content preview for debugging
        st.write(f"API Response status: {response.status_code}")
        st.write(f"API Response preview: {response.text[:100]}...")
        
        # Check if the response is valid
        if response.status_code != 200:
            st.write(f"API error: {response.status_code} - {response.text}")
            # Fall back to the predefined analysis
            return get_fallback_analysis_plan()
            
        analysis_plan = response.text
        
        # Try to parse as JSON
        try:
            return json.loads(analysis_plan)
        except json.JSONDecodeError as json_err:
            st.write(f"JSON parsing error: {str(json_err)}")
            
            # If not valid JSON, extract code blocks
            code_blocks = re.findall(r'```python(.*?)```', analysis_plan, re.DOTALL)
            if code_blocks:
                return {f"Analysis Block {i+1}": block.strip() for i, block in enumerate(code_blocks)}
                
            # If we can't parse the response at all
            st.write("Could not parse code blocks from the response")
            return get_fallback_analysis_plan()
    except requests.exceptions.RequestException as req_err:
        st.write(f"Request error: {str(req_err)}")
        return get_fallback_analysis_plan()
    except Exception as e:
        st.write(f"Unexpected error: {str(e)}")
        return get_fallback_analysis_plan()

def get_fallback_analysis_plan():
    """Return a fallback analysis plan when the API call fails"""
    st.write("Using fallback analysis plan")
    return {
        "Data Cleaning": """
# Handle missing values
df_clean = df.copy()
# Fill numerical missing values with median
for col in df_clean.select_dtypes(include=['int64', 'float64']).columns:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
# Fill categorical missing values with mode
for col in df_clean.select_dtypes(include=['object']).columns:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown")
        """,
        "Basic Statistics": """
# Get descriptive statistics for numerical columns
desc_stats = df.describe().T
desc_stats['missing'] = df.isnull().sum()
desc_stats['missing_percentage'] = (df.isnull().sum() / len(df)) * 100
        """,
        "Correlation Analysis": """
# Calculate correlation matrix for numerical columns
corr_matrix = df.select_dtypes(include=['int64', 'float64']).corr()

# Plot correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
        """,
        "Distribution Plots": """
# Plot histograms for numerical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns[:5]  # Limit to first 5
fig, axes = plt.subplots(len(num_cols), 1, figsize=(12, 4*len(num_cols)))
if len(num_cols) == 1:
    axes = [axes]
    
for i, col in enumerate(num_cols):
    sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')
plt.tight_layout()
        """,
        "Categorical Analysis": """
# Analyze categorical columns
cat_cols = df.select_dtypes(include=['object', 'category']).columns[:5]  # Limit to first 5
for col in cat_cols:
    plt.figure(figsize=(10, 6))
    value_counts = df[col].value_counts()
    sns.barplot(x=value_counts.index, y=value_counts.values)
    plt.xticks(rotation=45)
    plt.title(f'Value Counts for {col}')
    plt.tight_layout()
        """
    }

def execute_analysis_code(df, code_blocks):
    """Execute generated analysis code blocks and capture results and figures"""
    results = {}
    figures = []
    
    for name, code_block in code_blocks.items():
        try:
            # Set up locals dictionary with the dataframe
            local_vars = {"df": df, "plt": plt, "sns": sns, "np": np, "pd": pd}
            
            # Execute the code block
            exec(code_block, globals(), local_vars)
            
            # Capture any matplotlib figures
            if plt.get_fignums():
                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    img_buf = io.BytesIO()
                    fig.savefig(img_buf, format='png', bbox_inches='tight')
                    img_buf.seek(0)
                    figures.append((name, img_buf))
                plt.close('all')
            
            # Store other variables
            for var_name, var_value in local_vars.items():
                if var_name not in ["df", "plt", "sns", "np", "pd"] and not var_name.startswith("_"):
                    if isinstance(var_value, pd.DataFrame) and not var_value.empty and var_value.shape[0] < 100:
                        results[f"{name} - {var_name}"] = var_value
                    elif isinstance(var_value, (int, float, str, list, dict)):
                        results[f"{name} - {var_name}"] = var_value
                    
        except Exception as e:
            results[f"Error in {name}"] = str(e)
    
    return results, figures

def generate_final_report(data_info, analysis_results, figures):
    """Generate final analysis report using LLM"""
    # Prepare content for the LLM
    analysis_summary = {
        "data_info": {k: v for k, v in data_info.items() if k not in ["sample_data", "summary_stats"]},
        "analysis_results": {k: str(v)[:500] + "..." if len(str(v)) > 500 else str(v) 
                            for k, v in analysis_results.items() 
                            if isinstance(v, (int, float, str, list, dict))}
    }
    
    prompt = f"""
    Generate a comprehensive data analysis report based on the following information:
    {json.dumps(analysis_summary)}
    
    The report should include:
    1. Executive summary
    2. Data overview
    3. Key insights
    4. Detailed analysis of patterns and relationships
    5. Recommendations based on the data
    
    Format the report in markdown with appropriate headings and sections.
    """
    
    # In a real application, call the LLM API
    # For demo purposes, we'll use a predefined response
    try:
        response = requests.get(f"https://text.pollinations.ai/{prompt}", timeout=10)
        report = response.text
        return report
    except:
        # Fallback basic report
        return f"""
        # Data Analysis Report
        
        ## Executive Summary
        This report contains an analysis of a dataset with {data_info['shape'][0]} rows and {data_info['shape'][1]} columns.
        
        ## Data Overview
        - Number of records: {data_info['shape'][0]}
        - Number of features: {data_info['shape'][1]}
        - Numerical features: {', '.join(data_info['numerical_columns'])}
        - Categorical features: {', '.join(data_info['categorical_columns'])}
        - Missing values: {sum(data_info['missing_values'].values())} total
        
        ## Key Insights
        The analysis has identified several patterns and relationships in the data.
        Refer to the visualizations for detailed insights.
        """

if uploaded_file is not None:
    # Add debug information
    st.write(f"File name: {uploaded_file.name}")
    st.write(f"File size: {uploaded_file.size} bytes")
    st.write(f"File type: {uploaded_file.type}")
    
    # Show loading spinner
    with st.spinner("Reading and analyzing your data..."):
        # Read the file based on its type
        file_extension = uploaded_file.name.split('.')[-1].lower()
        try:
            st.write(f"Attempting to read file as {file_extension} format...")
            
            if file_extension == 'csv':
                # Add error handling parameters for CSV
                df = pd.read_csv(uploaded_file, on_bad_lines='skip', skipinitialspace=True)
                st.write(f"CSV read attempt complete. DataFrame shape: {df.shape if df is not None else 'None'}")
            elif file_extension == 'json':
                try:
                    # First try standard JSON
                    st.write("Trying standard JSON format...")
                    df = pd.read_json(uploaded_file)
                    st.write(f"Standard JSON read successful. DataFrame shape: {df.shape}")
                except Exception as e1:
                    st.write(f"Standard JSON read failed: {str(e1)}")
                    try:
                        # Reset file pointer before second attempt
                        uploaded_file.seek(0)
                        # Try JSON Lines format
                        st.write("Trying JSON Lines format...")
                        df = pd.read_json(uploaded_file, lines=True)
                        st.write(f"JSON Lines read successful. DataFrame shape: {df.shape}")
                    except Exception as e2:
                        st.write(f"JSON Lines read failed: {str(e2)}")
                        try:
                            # One more attempt - try reading as text first
                            uploaded_file.seek(0)
                            st.write("Trying to read JSON as text first...")
                            json_str = uploaded_file.read().decode('utf-8')
                            data = json.loads(json_str)
                            # Handle different JSON structures
                            if isinstance(data, list):
                                df = pd.DataFrame(data)
                            elif isinstance(data, dict):
                                if any(isinstance(data[k], list) for k in data):
                                    # Find the list in the dictionary
                                    for k in data:
                                        if isinstance(data[k], list):
                                            df = pd.DataFrame(data[k])
                                            break
                                else:
                                    # Singleton dictionary, make it a single row
                                    df = pd.DataFrame([data])
                            st.write(f"Custom JSON parsing successful. DataFrame shape: {df.shape}")
                        except Exception as e3:
                            st.write(f"All JSON reading methods failed: {str(e3)}")
                            df = None
            elif file_extension in ['xlsx', 'xls']:
                # Add error handling for Excel files
                df = pd.read_excel(uploaded_file, engine='openpyxl' if file_extension == 'xlsx' else 'xlrd')
            else:
                st.error("Unsupported file format")
                st.stop()

            # Validate that we got a dataframe
            if df is None or df.empty:
                st.error("The file appears to be empty or corrupted")
                st.stop()
                
            # Remove any trailing whitespace in column names
            df.columns = df.columns.str.strip()
            
            # Get initial data info
            data_info = get_initial_data_info(df)
            
            # Display initial data information
            st.subheader("Initial Data Overview")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Rows:** {data_info['shape'][0]}, **Columns:** {data_info['shape'][1]}")
                st.write("**Column Types:**")
                for col, dtype in data_info['dtypes'].items():
                    st.write(f"- {col}: {dtype}")
            
            with col2:
                st.write("**Missing Values:**")
                missing_df = pd.DataFrame.from_dict(data_info['missing_values'], orient='index', 
                                                 columns=['Count']).reset_index()
                missing_df.columns = ['Column', 'Missing Count']
                missing_df['Missing Percentage'] = missing_df['Missing Count'] / data_info['shape'][0] * 100
                st.dataframe(missing_df)
            
            # Display sample data
            st.subheader("Sample Data")
            st.dataframe(df.head())
            
            # Get analysis plan from LLM
            if st.button("Generate Detailed Analysis"):
                with st.spinner("Generating analysis plan..."):
                    analysis_plan = get_analysis_plan(data_info)
                
                if "error" in analysis_plan:
                    st.error(f"Error generating analysis plan: {analysis_plan['error']}")
                    st.stop()
                
                # Execute analysis code
                with st.spinner("Executing analysis..."):
                    results, figures = execute_analysis_code(df, analysis_plan)
                
                # Display results
                st.subheader("Analysis Results")
                
                # Display data frames from results
                for name, result in results.items():
                    if isinstance(result, pd.DataFrame):
                        st.write(f"**{name}**")
                        st.dataframe(result)
                    elif isinstance(result, str) and name.startswith("Error"):
                        st.error(f"{name}: {result}")
                    elif isinstance(result, (int, float, str, list, dict)):
                        st.write(f"**{name}**")
                        st.write(result)
                
                # Display figures
                if figures:
                    st.subheader("Visualizations")
                    # Create tabs for each figure
                    tabs = st.tabs([name for name, _ in figures])
                    for i, (tab, (name, fig_buffer)) in enumerate(zip(tabs, figures)):
                        with tab:
                            st.image(fig_buffer, caption=name)
                
                # Generate final report
                with st.spinner("Generating final report..."):
                    report = generate_final_report(data_info, results, figures)
                
                # Display final report
                st.subheader("Final Analysis Report")
                st.markdown(report)
                
                # Add download button for report
                report_html = f"""
                <html>
                <head>
                    <title>Data Analysis Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; }}
                        h1 {{ color: #2C3E50; }}
                        h2 {{ color: #3498DB; margin-top: 30px; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                    </style>
                </head>
                <body>
                    {report}
                </body>
                </html>
                """
                
                b64 = base64.b64encode(report_html.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64}" download="analysis_report.html">Download Report as HTML</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.stop()
else:
    # Display info when no file is uploaded
    st.info("Please upload a data file to begin analysis.")
    
    # Display demo information
    st.subheader("How it works")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**1. Upload Data**")
        st.write("Support for CSV, JSON, and Excel formats")
    
    with col2:
        st.write("**2. AI Analysis**")
        st.write("Automatic data cleaning, statistics, and visualization")
    
    with col3:
        st.write("**3. Get Report**")
        st.write("Comprehensive insights with downloadable report")