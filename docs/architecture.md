
flowchart TD
    A["M5 Walmart Dataset<br/>Sales, Calendar, Prices"]
    B["Data Validation and EDA"]
    C["Feature Engineering<br/>Lag, Rolling, Calendar, Price"]

    D["Seasonal Naive Baseline"]
    E["LightGBM<br/>Product-Store Forecast"]
    F["Prophet<br/>National Aggregate Forecast"]
    G["Temporal Fusion Transformer<br/>60-Series Experiment"]

    H["Hierarchical Reconciliation<br/>Store → State → National"]
    I["Prediction Interval Evaluation"]
    J["OR-Tools CP-SAT<br/>Inventory Optimization"]

    K["Evaluation Reports<br/>WAPE, MAE, RMSE"]
    L["Inventory Recommendations"]
    M["Plotly Dash Dashboard"]

    A --> B
    B --> C

    C --> D
    C --> E
    C --> G
    B --> F

    E --> H
    F --> I

    D --> K
    E --> K
    F --> K
    G --> K

    E --> J
    J --> L

    H --> M
    I --> M
    K --> M
    L --> M
