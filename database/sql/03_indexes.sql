-- =====================================================
-- FRAUD DETECTION SYSTEM
-- DATABASE INDEXING
-- PostgreSQL 18
-- =====================================================


-- 1. USER INDEXES

CREATE UNIQUE INDEX idx_users_email
ON users(email);


-- 2. ACCOUNT INDEXES

CREATE INDEX idx_accounts_user_id
ON accounts(user_id);


-- 3. MERCHANT INDEXES

CREATE INDEX idx_merchants_category
ON merchants(merchant_category);


-- 4. TRANSACTION INDEXES

CREATE INDEX idx_transactions_sender
ON transactions(sender_account_id);

CREATE INDEX idx_transactions_receiver
ON transactions(receiver_account_id);

CREATE INDEX idx_transactions_merchant
ON transactions(merchant_id);

CREATE INDEX idx_transactions_occurred_at
ON transactions(occurred_at DESC);

CREATE INDEX idx_transactions_status
ON transactions(transaction_status);

CREATE INDEX idx_transactions_sender_time
ON transactions(sender_account_id, occurred_at DESC);


-- 5. FRAUD ALERT INDEXES

CREATE INDEX idx_fraud_alerts_transaction
ON fraud_alerts(transaction_id);

CREATE INDEX idx_fraud_alerts_risk
ON fraud_alerts(risk_score DESC);

CREATE INDEX idx_fraud_alerts_status
ON fraud_alerts(alert_status);

CREATE INDEX idx_open_fraud_alerts
ON fraud_alerts(created_at DESC)
WHERE alert_status = 'open';


-- 6. AUDIT LOG INDEXES

CREATE INDEX idx_audit_transaction
ON transaction_audit_logs(transaction_id);

CREATE INDEX idx_audit_created_at
ON transaction_audit_logs(created_at DESC);


-- =====================================================
-- END OF INDEXING
-- =====================================================