CREATE TABLE IF NOT EXISTS customer_risk_scores (
    customer_id VARCHAR NOT NULL,
    scoring_date DATE NOT NULL,
    churn_risk_3m FLOAT NOT NULL,
    churn_risk_6m FLOAT NOT NULL,
    churn_risk_12m FLOAT NOT NULL,
    churn_risk_24m FLOAT NOT NULL,
    -- Restricted expected active month-ends through the configured CLV horizon (24).
    expected_remaining_lifetime FLOAT NOT NULL,
    predicted_clv FLOAT NOT NULL,
    top_risk_driver VARCHAR NOT NULL,
    model_version VARCHAR NOT NULL,
    -- UTC timestamp; TIMESTAMP is portable across DuckDB and Snowflake.
    scored_at TIMESTAMP NOT NULL,
    PRIMARY KEY (customer_id, scoring_date)
);
