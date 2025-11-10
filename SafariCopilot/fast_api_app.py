from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import Optional
import asyncpg
import redis.asyncio as redis
from contextlib import asynccontextmanager
import os
from datetime import datetime
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment variables
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_USER = os.getenv("PG_USER", "app")
PG_PASSWORD = os.getenv("PG_PASSWORD", "app")
PG_DATABASE = os.getenv("PG_DATABASE", "copilot")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Global connection pools
pg_pool: Optional[asyncpg.Pool] = None
redis_client: Optional[redis.Redis] = None


# Pydantic models
class StartCallRequest(BaseModel):
    call_id: str
    begin_ts: datetime


class EndCallRequest(BaseModel):
    call_id: str
    end_ts: datetime
    call_summary: str


class AddCallChunkRequest(BaseModel):
    call_id: str
    participant: str
    begin_ts: datetime
    end_ts: datetime
    text: str


class UpdateSentimentRequest(BaseModel):
    call_id: str
    sentiment: str


class UpdateRecommendationRequest(BaseModel):
    call_id: str
    recommendation: str


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global pg_pool, redis_client

    try:
        # Initialize PostgreSQL connection pool
        pg_pool = await asyncpg.create_pool(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASSWORD,
            database=PG_DATABASE,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        logger.info("PostgreSQL connection pool created")

        # Initialize Redis connection pool
        redis_client = await redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            db=REDIS_DB,
            decode_responses=True,
            max_connections=20
        )
        await redis_client.ping()
        logger.info("Redis connection established")

    except Exception as e:
        logger.error(f"Failed to initialize connections: {e}")
        raise

    yield

    # Shutdown
    if pg_pool:
        await pg_pool.close()
        logger.info("PostgreSQL connection pool closed")

    if redis_client:
        await redis_client.close()
        logger.info("Redis connection closed")


app = FastAPI(title="Copilot Data API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check PostgreSQL
        async with pg_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")

        # Check Redis
        await redis_client.ping()

        return {"status": "healthy", "postgresql": "connected", "redis": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@app.post("/init_pg_extensions")
async def init_pg_extensions():
    """
    Initialize PostgreSQL extensions required for partition management.
    This should be run once before creating tables.
    Requires superuser privileges.
    """
    try:
        async with pg_pool.acquire() as conn:
            # Check and create pg_partman extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_partman;")
            logger.info("pg_partman extension created/verified")

            # Check and create pg_cron extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_cron;")
            logger.info("pg_cron extension created/verified")

            return {
                "status": "success",
                "message": "PostgreSQL extensions initialized successfully",
                "extensions": ["pg_partman", "pg_cron"]
            }
    except asyncpg.exceptions.InsufficientPrivilegeError as e:
        logger.error(f"Insufficient privileges to create extensions: {e}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient privileges. Superuser access required to create extensions."
        )
    except Exception as e:
        logger.error(f"Failed to initialize extensions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize extensions: {str(e)}"
        )


@app.post("/init_pg_table")
async def init_pg_table():
    """
    Create the calls table with hourly partitioning if it doesn't exist.
    Also sets up automatic partition management using pg_partman.
    """
    try:
        async with pg_pool.acquire() as conn:
            async with conn.transaction():
                # Create the partitioned table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS calls (
                        call_id TEXT NOT NULL,
                        begin_ts TIMESTAMP NOT NULL,
                        end_ts TIMESTAMP,
                        call_summary TEXT,
                        PRIMARY KEY (call_id, begin_ts)
                    ) PARTITION BY RANGE (begin_ts);
                """)
                logger.info("Calls table created/verified")

                # Create index on begin_ts
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_calls_begin_ts 
                    ON calls (begin_ts);
                """)
                logger.info("Index on begin_ts created/verified")

                # Check if partition management is already configured
                partition_exists = await conn.fetchval("""
                    SELECT EXISTS(
                        SELECT 1 FROM partman.part_config 
                        WHERE parent_table = 'public.calls'
                    );
                """)

                if not partition_exists:
                    # Configure automatic partition management
                    await conn.execute("""
                        SELECT partman.create_parent(
                            p_parent_table := 'public.calls',
                            p_control := 'begin_ts',
                            p_type := 'native',
                            p_interval := '1 hour',
                            p_premake := 24
                        );
                    """)
                    logger.info("Partition management configured")

                    # Update configuration
                    await conn.execute("""
                        UPDATE partman.part_config
                        SET infinite_time_partitions = true,
                            retention = '30 days',
                            retention_keep_table = false
                        WHERE parent_table = 'public.calls';
                    """)
                    logger.info("Partition configuration updated")

                    # Schedule automatic maintenance
                    cron_exists = await conn.fetchval("""
                        SELECT EXISTS(
                            SELECT 1 FROM cron.job 
                            WHERE jobname = 'partman-maintenance'
                        );
                    """)

                    if not cron_exists:
                        await conn.execute("""
                            SELECT cron.schedule(
                                'partman-maintenance',
                                '0 * * * *',
                                $$SELECT partman.run_maintenance('public.calls')$$
                            );
                        """)
                        logger.info("Partition maintenance scheduled")

                return {
                    "status": "success",
                    "message": "Calls table and partitioning configured successfully",
                    "partition_management": "configured" if not partition_exists else "already_exists"
                }

    except Exception as e:
        logger.error(f"Failed to initialize table: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize table: {str(e)}"
        )


@app.post("/start_call", status_code=status.HTTP_201_CREATED)
async def start_call(request: StartCallRequest):
    """
    Insert a new call to the calls table with call_id and begin_ts.
    """
    try:
        async with pg_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO calls (call_id, begin_ts)
                VALUES ($1, $2)
            """, request.call_id, request.begin_ts)

        logger.info(f"Call started: {request.call_id}")
        return {
            "status": "success",
            "message": "Call started successfully",
            "call_id": request.call_id,
            "begin_ts": request.begin_ts.isoformat()
        }

    except asyncpg.exceptions.UniqueViolationError:
        logger.warning(f"Duplicate call_id attempted: {request.call_id}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Call with ID {request.call_id} already exists"
        )
    except Exception as e:
        logger.error(f"Failed to start call: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start call: {str(e)}"
        )


@app.put("/end_call")
async def end_call(request: EndCallRequest):
    """
    Update an existing call with end_ts and call_summary.
    """
    try:
        async with pg_pool.acquire() as conn:
            result = await conn.execute("""
                UPDATE calls
                SET end_ts = $2, call_summary = $3
                WHERE call_id = $1
            """, request.call_id, request.end_ts, request.call_summary)

            # Check if any row was updated
            if result == "UPDATE 0":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Call with ID {request.call_id} not found"
                )

        logger.info(f"Call ended: {request.call_id}")
        return {
            "status": "success",
            "message": "Call ended successfully",
            "call_id": request.call_id,
            "end_ts": request.end_ts.isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to end call: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to end call: {str(e)}"
        )


@app.post("/add_call_chunk", status_code=status.HTTP_201_CREATED)
async def add_call_chunk(request: AddCallChunkRequest):
    """
    Add a new chunk to Redis stream for the specified call.
    """
    try:
        stream_key = f"call:{request.call_id}"

        # Prepare chunk data
        chunk_data = {
            "participant": request.participant,
            "begin_ts": request.begin_ts.isoformat(),
            "end_ts": request.end_ts.isoformat(),
            "text": request.text
        }

        # Add to Redis stream (XADD with auto-generated ID)
        stream_id = await redis_client.xadd(stream_key, chunk_data)

        logger.info(f"Chunk added to call {request.call_id}: {stream_id}")
        return {
            "status": "success",
            "message": "Call chunk added successfully",
            "call_id": request.call_id,
            "stream_id": stream_id
        }

    except Exception as e:
        logger.error(f"Failed to add call chunk: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add call chunk: {str(e)}"
        )


@app.put("/update_sentiment")
async def update_sentiment(request: UpdateSentimentRequest):
    """
    Update or set the sentiment for a call in Redis analytics.
    """
    try:
        analytics_key = f"call:{request.call_id}:analytics"

        # Update sentiment in the analytics hash
        await redis_client.hset(analytics_key, "sentiment", request.sentiment)

        logger.info(f"Sentiment updated for call {request.call_id}")
        return {
            "status": "success",
            "message": "Sentiment updated successfully",
            "call_id": request.call_id,
            "sentiment": request.sentiment
        }

    except Exception as e:
        logger.error(f"Failed to update sentiment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update sentiment: {str(e)}"
        )


@app.put("/update_recommendation")
async def update_recommendation(request: UpdateRecommendationRequest):
    """
    Update or set the recommendation for a call in Redis analytics.
    """
    try:
        analytics_key = f"call:{request.call_id}:analytics"

        # Update recommendation in the analytics hash
        await redis_client.hset(analytics_key, "recommendation", request.recommendation)

        logger.info(f"Recommendation updated for call {request.call_id}")
        return {
            "status": "success",
            "message": "Recommendation updated successfully",
            "call_id": request.call_id,
            "recommendation": request.recommendation
        }

    except Exception as e:
        logger.error(f"Failed to update recommendation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update recommendation: {str(e)}"
        )


# Optional: Utility endpoints for debugging/monitoring
@app.get("/call/{call_id}/chunks")
async def get_call_chunks(call_id: str, count: int = 100):
    """
    Retrieve chunks from Redis stream for a specific call.
    """
    try:
        stream_key = f"call:{call_id}"

        # Check if stream exists
        exists = await redis_client.exists(stream_key)
        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No chunks found for call {call_id}"
            )

        # Read from stream
        chunks = await redis_client.xrange(stream_key, count=count)

        return {
            "call_id": call_id,
            "chunk_count": len(chunks),
            "chunks": [
                {"stream_id": chunk_id, "data": data}
                for chunk_id, data in chunks
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve chunks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chunks: {str(e)}"
        )


@app.get("/call/{call_id}/analytics")
async def get_call_analytics(call_id: str):
    """
    Retrieve analytics for a specific call from Redis.
    """
    try:
        analytics_key = f"call:{call_id}:analytics"

        # Get all fields from analytics hash
        analytics = await redis_client.hgetall(analytics_key)

        if not analytics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No analytics found for call {call_id}"
            )

        return {
            "call_id": call_id,
            "analytics": analytics
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve analytics: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)