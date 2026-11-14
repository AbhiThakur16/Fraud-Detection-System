-- =====================================================
-- PROJECT: REAL-TIME FRAUD DETECTION SYSTEM
-- DATABASE: fraud_detection_db
-- FILE: 01_create_tables.sql
-- =====================================================


-- 1. USERS TABLE
CREATE TABLE users (
    user_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    full_name VARCHAR(120) NOT NULL,
    email VARCHAR(255) NOT NULL,
    phone VARCHAR(20),

    country_code CHAR(2) DEFAULT 'IN',

    is_active BOOLEAN DEFAULT TRUE,

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT chk_user_name
        CHECK (LENGTH(TRIM(full_name)) > 0)
);


-- 2. ACCOUNTS TABLE
CREATE TABLE accounts (
    account_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    user_id BIGINT NOT NULL,

    account_number VARCHAR(34) NOT NULL UNIQUE,

    account_type VARCHAR(20) NOT NULL,

    currency CHAR(3) DEFAULT 'INR',

    account_status VARCHAR(20) DEFAULT 'active',

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_account_user
        FOREIGN KEY (user_id)
        REFERENCES users(user_id)
        ON DELETE RESTRICT,

    CONSTRAINT chk_account_type
        CHECK (account_type IN (
            'savings',
            'current',
            'credit'
        )),

    CONSTRAINT chk_account_status
        CHECK (account_status IN (
            'active',
            'blocked',
            'closed'
        ))
);


-- 3. MERCHANTS TABLE
CREATE TABLE merchants (
    merchant_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    merchant_name VARCHAR(150) NOT NULL,

    merchant_code VARCHAR(50) UNIQUE NOT NULL,

    merchant_category VARCHAR(80),

    country_code CHAR(2) DEFAULT 'IN',

    is_active BOOLEAN DEFAULT TRUE,

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);


-- 4. TRANSACTIONS TABLE
CREATE TABLE transactions (
    transaction_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    external_reference VARCHAR(100) UNIQUE NOT NULL,

    sender_account_id BIGINT NOT NULL,

    receiver_account_id BIGINT,

    merchant_id BIGINT,

    transaction_type VARCHAR(30) NOT NULL,

    transaction_channel VARCHAR(30) NOT NULL,

    amount NUMERIC(18,2) NOT NULL,

    currency CHAR(3) DEFAULT 'INR',

    transaction_status VARCHAR(20) DEFAULT 'pending',

    is_fraud BOOLEAN DEFAULT NULL,

    occurred_at TIMESTAMPTZ NOT NULL,

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_transaction_sender
        FOREIGN KEY (sender_account_id)
        REFERENCES accounts(account_id)
        ON DELETE RESTRICT,

    CONSTRAINT fk_transaction_receiver
        FOREIGN KEY (receiver_account_id)
        REFERENCES accounts(account_id)
        ON DELETE RESTRICT,

    CONSTRAINT fk_transaction_merchant
        FOREIGN KEY (merchant_id)
        REFERENCES merchants(merchant_id)
        ON DELETE RESTRICT,

    CONSTRAINT chk_transaction_amount
        CHECK (amount > 0),

    CONSTRAINT chk_transaction_type
        CHECK (transaction_type IN (
            'purchase',
            'transfer',
            'withdrawal',
            'deposit',
            'refund'
        )),

    CONSTRAINT chk_transaction_channel
        CHECK (transaction_channel IN (
            'card',
            'upi',
            'net_banking',
            'atm',
            'mobile',
            'web'
        )),

    CONSTRAINT chk_transaction_status
        CHECK (transaction_status IN (
            'pending',
            'authorized',
            'settled',
            'declined',
            'reversed'
        )),

    CONSTRAINT chk_different_accounts
        CHECK (
            receiver_account_id IS NULL
            OR receiver_account_id <> sender_account_id
        )
);


-- 5. FRAUD ALERTS TABLE
CREATE TABLE fraud_alerts (
    alert_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    transaction_id BIGINT NOT NULL,

    model_name VARCHAR(100) NOT NULL,

    model_version VARCHAR(50),

    risk_score NUMERIC(5,4) NOT NULL,

    severity VARCHAR(20) NOT NULL,

    alert_reason JSONB DEFAULT '{}'::JSONB,

    alert_status VARCHAR(30) DEFAULT 'open',

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    reviewed_at TIMESTAMPTZ,

    CONSTRAINT fk_alert_transaction
        FOREIGN KEY (transaction_id)
        REFERENCES transactions(transaction_id)
        ON DELETE RESTRICT,

    CONSTRAINT chk_risk_score
        CHECK (risk_score BETWEEN 0 AND 1),

    CONSTRAINT chk_alert_severity
        CHECK (severity IN (
            'low',
            'medium',
            'high',
            'critical'
        )),

    CONSTRAINT chk_alert_status
        CHECK (alert_status IN (
            'open',
            'reviewed',
            'confirmed_fraud',
            'false_positive',
            'closed'
        ))
);


-- 6. TRANSACTION AUDIT LOGS TABLE
CREATE TABLE transaction_audit_logs (
    audit_log_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    transaction_id BIGINT NOT NULL,

    event_type VARCHAR(80) NOT NULL,

    actor_type VARCHAR(20) NOT NULL,

    actor_id BIGINT,

    event_data JSONB DEFAULT '{}'::JSONB,

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_audit_transaction
        FOREIGN KEY (transaction_id)
        REFERENCES transactions(transaction_id)
        ON DELETE RESTRICT,

    CONSTRAINT chk_actor_type
        CHECK (actor_type IN (
            'system',
            'user',
            'admin'
        ))
);


-- =====================================================
-- END OF DATABASE SCHEMA
-- =====================================================