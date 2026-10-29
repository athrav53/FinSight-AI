-- FinSight AI database schema (PostgreSQL / Neon)

CREATE TABLE IF NOT EXISTS bank_transactions (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(30),
    account_number VARCHAR(30),
    transaction_date DATE NOT NULL,
    transaction_type VARCHAR(10) NOT NULL
        CHECK (transaction_type IN ('Debit', 'Credit')),
    category VARCHAR(100),
    merchant VARCHAR(200),
    description TEXT,
    amount NUMERIC(15, 2) NOT NULL
        CHECK (amount > 0),
    balance NUMERIC(15, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_bank_transactions_user_date
    ON bank_transactions (user_id, transaction_date);