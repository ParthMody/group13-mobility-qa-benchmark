# Task 1: Next-POI Category QA

## Goal

Given a user's previous POI check-in sequence, answer the most likely next POI category.

## Raw Check-In Format

Raw check-in CSV files must include:

- `user_id`
- `venue_id`
- `venue_category`
- `timestamp`
- `venue_city`
- `latitude`
- `longitude`

## Generated QA Format

Generated examples are stored as JSONL records. Each line is one QA example with context, choices, answer, and metadata.

## Current Scope

This phase defines the data shape and validation only. It does not include model training or benchmark evaluation.
