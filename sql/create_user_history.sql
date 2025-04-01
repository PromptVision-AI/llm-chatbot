CREATE TABLE chat_history (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    message TEXT NOT NULL,
    response TEXT NOT NULL,
    image_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create an index on user_id for faster queries
CREATE INDEX idx_chat_history_user_id ON chat_history(user_id);

-- Create an index on timestamp for faster sorting
CREATE INDEX idx_chat_history_timestamp ON chat_history(timestamp);
