# Quick Start

Follow these steps to run the application:

1. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Fetch/Update Data (If Necessary):**  
   Ensure the latest CSV data files (e.g., `ahmedabad.csv`) are in the `app/data/` directory. If needed, run the script to fetch data:
    ```bash
    python "app\data\data.py"
    ```

3. **Train Models (Required):**  
   Before starting the server, train the models. This creates the necessary `.pkl` files in `app/models/`.
    ```bash
    python -m app.ml.train_models
    ```
    *(This step can take some time.)*

4. **Start Backend Server:**
    ```bash
    uvicorn app.main:app --reload
    ```
    Keep this terminal open. The API runs at `http://127.0.0.1:8000`.

5. **Open Frontend:**  
   Open the `frontend/index.html` file in your web browser using Live Server:
   - In VS Code, right-click on the file
   - Select "Open with Live Server"
   - The page will open in your default browser

---

## API Endpoint
- **GET `/predict`**
    - **Query Parameters:**
        - `city` (str, required): e.g., `ahmedabad`
        - `model_name` (str, required): e.g., `LightGBM`, `Ensemble`, `Prophet`
        - `forecast_type` (str, required): `48h`, `1week`, `2weeks`
        - `day_of_week` (int, optional): 0-6 (Mon-Sun), only for `1week` or `2weeks` type.
        - `prophet_extended` (str, optional): `1month`, `3months`, `6months`, `1year` (Prophet model only)
        - `include_bounds` (bool, optional): Include prediction uncertainty bounds (Prophet model only)
    - **Returns:** JSON array of forecast objects.

## Prophet Extended Forecasting

The Prophet model allows for longer-term forecasting beyond the standard durations:

- Select "Prophet (Meta)" in the model dropdown
- Choose an extended forecast period (up to 1 year)
- Optionally enable uncertainty bounds to see upper and lower prediction intervals
