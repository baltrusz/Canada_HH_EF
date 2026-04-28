# AI Coding Guidelines for Canada Household Energy Project

## Project Overview
This is a Python-based data analysis project focused on Canadian household energy consumption data. The codebase currently consists of data loading and processing scripts.

## Key Patterns and Conventions

### Data Handling
- Use pandas for data manipulation and analysis
- Read SAS data files using `pd.read_sas()` for household spending survey data
- Handle Windows file paths using raw strings (e.g., `r"C:\path\to\file"`) to avoid escape issues

### File Structure
- Main scripts are in the root directory (e.g., `coding_test.py`)
- Data files are located in external directories referenced by absolute paths

### Dependencies
- Primary dependency: pandas for data processing
- Ensure SAS7BDAT files are accessible at specified paths

### Example Usage
```python
import pandas as pd
df = pd.read_sas(r"C:\path\to\data.sas7bdat")
```

## Development Workflow
- Run scripts directly with Python interpreter
- No build system currently in place
- Data files are external and not included in repository

Reference: [coding_test.py](coding_test.py) for basic data loading pattern.