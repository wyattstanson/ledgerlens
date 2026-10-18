-- Scaffold model so dbt, Dagster and their lineage wiring are exercised end to end
-- before any pack models exist. The audit schema itself is created by ledgerlens.core.db, not dbt;
-- dbt views over it arrive with the Audit and cost page in Phase 5, when this model is removed.
select
    'ledgerlens' as project,
    '0.1.0' as dbt_project_version
