-- Scaffold model so dbt, Dagster and their lineage wiring are exercised end to end
-- before any pack models exist. Replaced by audit and lineage views in Phase 1.
select
    'ledgerlens' as project,
    '0.1.0' as dbt_project_version
