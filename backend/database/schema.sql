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

-- Drop NOT NULL only if the column exists and is NOT NULL
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='users' AND column_name='password' AND is_nullable = 'NO'
    ) THEN
        ALTER TABLE users ALTER COLUMN password DROP NOT NULL;
    END IF;
END$$;

-- Add auth_provider column if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='users' AND column_name='auth_provider'
    ) THEN
        ALTER TABLE users ADD COLUMN auth_provider TEXT DEFAULT 'email';
    END IF;
END$$;

-- Add google_uid column if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='users' AND column_name='google_uid'
    ) THEN
        ALTER TABLE users ADD COLUMN google_uid TEXT UNIQUE;
    END IF;
END$$;


-- Add password_null_check constraint if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints 
        WHERE constraint_name='password_null_check' AND table_name='users'
    ) THEN
        ALTER TABLE users ADD CONSTRAINT password_null_check
        CHECK (
            (auth_provider = 'google' AND password IS NULL)
            OR
            (auth_provider = 'email' AND password IS NOT NULL)
        );
    END IF;
END$$;

-- Add file_url column to results if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='results' AND column_name='file_url'
    ) THEN
        ALTER TABLE results ADD COLUMN file_url TEXT;
    END IF;
END$$;

-- Add file_url column to uploads if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='uploads' AND column_name='file_url'
    ) THEN
        ALTER TABLE uploads ADD COLUMN file_url TEXT;
    END IF;
END$$;

-- Add user_id to uploads if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='uploads' AND column_name='user_id'
    ) THEN
        ALTER TABLE uploads ADD COLUMN user_id INTEGER REFERENCES users(id);
    END IF;
END$$;

-- Add file_size to results if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='results' AND column_name='file_size'
    ) THEN
        ALTER TABLE results ADD COLUMN file_size BIGINT;
    END IF;
END$$;

-- Add file_size to uploads if not exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name='uploads' AND column_name='file_size'
    ) THEN
        ALTER TABLE uploads ADD COLUMN file_size BIGINT;
    END IF;
END$$;
