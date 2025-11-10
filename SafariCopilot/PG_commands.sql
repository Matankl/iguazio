-- Create the database (run only once)
CREATE DATABASE copilot;

-- Connect to the copilot database
\c copilot

-- Create the Calls table with hourly partitioning
-- Primary key must include the partition key (begin_ts)
CREATE TABLE calls (
    call_id TEXT NOT NULL,
    begin_ts TIMESTAMP NOT NULL,
    end_ts TIMESTAMP,
    call_summary TEXT,
    PRIMARY KEY (call_id, begin_ts)
) PARTITION BY RANGE (begin_ts);





-------- this part is optional but may lead to better performance
-- Create an index on begin_ts for better query performance
CREATE INDEX idx_calls_begin_ts ON calls (begin_ts);

-- Create a unique index on just call_id if needed
CREATE UNIQUE INDEX idx_calls_call_id ON calls (call_id);




---------------------- Schedule maintenance to the partitions
--installation
--# Ubuntu/Debian
--sudo apt-get install postgresql-14-cron  # Replace 14 with PG version

--# macOS (Homebrew)
--brew install pg_cron

-- import extension pg_partman
CREATE EXTENSION pg_partman;

-- Configure automatic partition management
SELECT partman.create_parent(
    p_parent_table := 'public.calls',
    p_control := 'begin_ts',
    p_type := 'native',
    p_interval := '1 hour',
    p_premake := 24  -- Create 24 hours of partitions in advance
);

-- Update the configuration for automatic maintenance
UPDATE partman.part_config
SET infinite_time_partitions = true,
    retention = '30 days',  -- optional: auto-drop partitions older than 30 days
    retention_keep_table = false
WHERE parent_table = 'public.calls';

-- Set up automatic maintenance (run this periodically)
SELECT partman.run_maintenance('public.calls');

-- activete run_maintenance periodically every 1H   (have to install the package!)
CREATE EXTENSION pg_cron;

SELECT cron.schedule(
    'partman-maintenance',
    '0 * * * *',  -- Every hour
    $$SELECT partman.run_maintenance('public.calls')$$
);