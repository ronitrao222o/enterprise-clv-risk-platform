-- Snowflake-compatible source contract. Use your configured database/schema.
-- Dates are month-end facts. Array EOS is the scheduled date known at install.
-- Foreign-key relationships are checked in src/data_quality.py, allowing atomic
-- replacement of local snapshot tables. Snowflake standard-table FKs are informational.
CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR NOT NULL PRIMARY KEY,
    industry VARCHAR NOT NULL,
    region VARCHAR NOT NULL,
    company_size VARCHAR NOT NULL,
    account_start_date DATE NOT NULL
);
CREATE TABLE IF NOT EXISTS arrays (
    array_id VARCHAR NOT NULL PRIMARY KEY,
    customer_id VARCHAR NOT NULL,
    product_family VARCHAR NOT NULL,
    install_date DATE NOT NULL,
    capacity_tb FLOAT NOT NULL,
    end_of_service_date DATE NOT NULL
);
CREATE TABLE IF NOT EXISTS monthly_account_metrics (
    customer_id VARCHAR NOT NULL,
    month DATE NOT NULL,
    storage_utilization FLOAT NOT NULL,
    data_growth FLOAT NOT NULL,
    support_tickets INTEGER NOT NULL,
    critical_incidents INTEGER NOT NULL,
    ARR FLOAT NOT NULL,
    expansion_amount FLOAT NOT NULL,
    PRIMARY KEY (customer_id, month)
);
CREATE TABLE IF NOT EXISTS events (
    customer_id VARCHAR NOT NULL,
    array_id VARCHAR,
    event_date DATE NOT NULL,
    event_type VARCHAR NOT NULL
);
