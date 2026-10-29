
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
SELECT * FROM bank_transactions;

SELECT
    COUNT(*) AS total_transactions,
    SUM(amount) AS total_spending
FROM bank_transactions
WHERE transaction_type = 'Debit';

SELECT
    category,
    ROUND(SUM(amount), 2) AS total_spent
FROM bank_transactions
WHERE transaction_type = 'Debit'
GROUP BY category
ORDER BY total_spent DESC;

SELECT
    category,
    ROUND(SUM(amount), 2) AS total_spent
FROM bank_transactions
WHERE transaction_type = 'Debit'
GROUP BY category
ORDER BY total_spent DESC;

SELECT
    merchant,
    COUNT(*) AS transaction_count,
    ROUND(SUM(amount), 2) AS total_spent
FROM bank_transactions
WHERE transaction_type = 'Debit'
  AND merchant IS NOT NULL
  AND LOWER(merchant) <> 'unknown'
GROUP BY merchant
ORDER BY total_spent DESC
LIMIT 10;

SELECT
    user_id,
    COUNT(*) AS transaction_count,
    ROUND(SUM(amount), 2) AS total_spending
FROM bank_transactions
WHERE transaction_type = 'Debit'
GROUP BY user_id
ORDER BY total_spending DESC;
