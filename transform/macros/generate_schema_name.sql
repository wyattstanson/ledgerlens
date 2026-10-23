{# A model's custom schema is used as-is ("fpa", not "main_fpa"), so dbt marts sit beside the tables the
   Python pipelines write to the same schema. #}
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is none -%}{{ target.schema }}{%- else -%}{{ custom_schema_name | trim }}{%- endif -%}
{%- endmacro %}
