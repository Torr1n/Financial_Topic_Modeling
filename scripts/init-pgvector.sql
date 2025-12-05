-- init-pgvector.sql
-- Enable pgvector extension for vector similarity search
-- This script runs automatically when the container starts

CREATE EXTENSION IF NOT EXISTS vector;
