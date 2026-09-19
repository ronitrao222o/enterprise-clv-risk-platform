-- Executable in both DuckDB and Snowflake after python -m src.score.
SELECT
    f.company_size,
    f.product_family,
    COUNT(*) AS active_accounts,
    SUM(f.ARR) AS total_arr,
    SUM(s.predicted_clv) AS predicted_24m_contribution,
    SUM(f.ARR * s.churn_risk_12m) AS arr_risk_proxy,
    AVG(s.churn_risk_12m) AS mean_12m_churn_risk
FROM customer_risk_scores s
JOIN account_features f
  ON s.customer_id = f.customer_id AND s.scoring_date = f.scoring_date
GROUP BY f.company_size, f.product_family
ORDER BY arr_risk_proxy DESC;
