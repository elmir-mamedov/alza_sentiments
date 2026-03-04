-- 1. Create the table
CREATE TABLE IF NOT EXISTS reviews (
    id    SERIAL PRIMARY KEY,
    text  TEXT,
    label INTEGER
);

-- 2. Load the CSV
\copy reviews(text, label) FROM '/Users/elmirmamedov/Documents/API_prj/get_data/batch_of_reviews.csv' CSV HEADER;

-- 3. Verify
SELECT COUNT(*)                                    AS total,
       SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS positives,
       SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) AS negatives
FROM reviews;