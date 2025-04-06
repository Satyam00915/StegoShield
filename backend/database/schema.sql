-- Users table updated with more useful fields
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Results table (unchanged as per your instruction)
CREATE TABLE IF NOT EXISTS results (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    prediction TEXT NOT NULL,
    confidence FLOAT NOT NULL,
    user_id INTEGER REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS uploads (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    filetype TEXT NOT NULL,
    result TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- Optional: Suspicious file reports
CREATE TABLE IF NOT EXISTS suspicious_reports (
    id SERIAL PRIMARY KEY,
    upload_id INTEGER REFERENCES results(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id),
    reason TEXT,
    report_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved BOOLEAN DEFAULT FALSE
);

-- Optional: Blog section (for admin-posted blogs)
CREATE TABLE IF NOT EXISTS blog_posts (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    author_id INTEGER REFERENCES users(id),
    published_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Optional: Feedback form
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    message TEXT NOT NULL,
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE users ALTER COLUMN password DROP NOT NULL;

ALTER TABLE users ADD COLUMN auth_provider TEXT DEFAULT 'email';

ALTER TABLE users ADD CONSTRAINT password_null_check
CHECK (
    (auth_provider = 'google' AND password IS NULL)
    OR
    (auth_provider = 'email' AND password IS NOT NULL)
);
