CREATE TABLE IF NOT EXISTS prediction_history (

    id BIGSERIAL PRIMARY KEY,

    transaction_id UUID UNIQUE NOT NULL,

    transaction_amount NUMERIC(15, 2) NOT NULL,

    fraud_probability NUMERIC(7, 4) NOT NULL,

    prediction SMALLINT NOT NULL,

    status VARCHAR(30) NOT NULL,

    threshold NUMERIC(5, 4) NOT NULL,

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT chk_prediction
        CHECK (prediction IN (0, 1)),

    CONSTRAINT chk_fraud_probability
        CHECK (
            fraud_probability >= 0
            AND fraud_probability <= 100
        ),

    CONSTRAINT chk_threshold
        CHECK (threshold >= 0 AND threshold <= 1)

);

CREATE INDEX IF NOT EXISTS idx_prediction_created_at
ON prediction_history(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_prediction_status
ON prediction_history(status);