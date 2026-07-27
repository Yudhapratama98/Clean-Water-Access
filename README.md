# Clean Water Access (Akses Air Bersih)

An end-to-end data science and web application project to analyze and model clean water access in Indonesia. This project integrates data clustering, predictive modeling, and interactive data visualization.

## Features
- **Machine Learning**: Predictive modeling using scikit-learn/XGBoost to forecast or classify clean water access levels.
- **Data Analysis**: Jupyter Notebooks for data preprocessing, EDA, clustering, and modeling.
- **Web Interface**: A Flask-based web application for serving predictions and interacting with the machine learning models.
- **Geospatial Data**: Includes GADM shapefiles for precise Indonesian regional mapping.
- **Tableau Dashboards**: Data visualization and business intelligence via a dedicated Tableau workbook.

## Project Structure
- `app.py`: Main Flask application entry point.
- `static/`, `templates/`: Assets and HTML templates for the web interface.
- `Klaster/`: Output models and configurations from data clustering.
- `tableau/`: Contains Tableau workbook visualizations (`AksesAirBersih.twbx`).
- `gadm36_IDN_shp/`: Geospatial boundaries for mapping.
- `*.csv`: Cleaned and processed datasets used for modeling.
- `*.ipynb`: Jupyter notebooks detailing the model training and exploration processes.

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Yudhapratama98/Clean-Water-Access.git
   cd Clean-Water-Access
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Run the application:**
   Ensure all dependencies are met, then launch the Flask server:
   ```bash
   python app.py
   ```
   Access the web app typically at `http://127.0.0.1:5000/`.
