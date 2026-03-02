-- extract_commodity_id.sql
-- Add column if it doesn't exist
ALTER TABLE url_queue
ADD COLUMN IF NOT EXISTS commodity_id INTEGER;

-- Populate only missing rows
UPDATE url_queue
SET commodity_id = COALESCE(
    (regexp_match(url, '-d([0-9]+)\.htm'))[1],
    (regexp_match(url, 'dq=([0-9]+)'))[1]
)::int
WHERE commodity_id IS NULL;

-- Optional: print number of missing rows after update
SELECT COUNT(*) AS missing_after_update
FROM url_queue
WHERE commodity_id IS NULL;
