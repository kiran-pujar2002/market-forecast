CREATE TABLE IF NOT EXISTS prices (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume BIGINT,
    UNIQUE (ticker, date)
);

CREATE TABLE IF NOT EXISTS news (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    published_at TIMESTAMP NOT NULL,
    headline TEXT NOT NULL,
    source VARCHAR(100),
    sentiment_score NUMERIC,
    sentiment_label VARCHAR(10)
);