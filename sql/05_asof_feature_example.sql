-- Portable SQL example of a leakage-safe monthly feature window.
-- Change only this literal to the desired month-end; never use CURRENT_DATE for backtests.
WITH parameters AS (
    SELECT CAST('2024-12-31' AS DATE) AS scoring_date
), active AS (
    SELECT m.customer_id, m.ARR, p.scoring_date
    FROM monthly_account_metrics m
    CROSS JOIN parameters p
    WHERE m.month = p.scoring_date
), support_history AS (
    SELECT m.customer_id,
           SUM(m.support_tickets) AS support_tickets_90d,
           SUM(m.critical_incidents) AS critical_incidents_90d
    FROM monthly_account_metrics m
    CROSS JOIN parameters p
    WHERE m.month <= p.scoring_date
      AND DATEDIFF('month', m.month, p.scoring_date) BETWEEN 0 AND 2
    GROUP BY m.customer_id
)
SELECT a.customer_id, a.scoring_date, a.ARR,
       h.support_tickets_90d, h.critical_incidents_90d
FROM active a
JOIN support_history h ON a.customer_id = h.customer_id;
