# SQL/PostgreSQL Cheatsheet - Interview Prep

**Complete reference for SQL interviews covering PostgreSQL syntax and standard SQL concepts**

---

## Table of Contents
1. [Data Types](#data-types)
2. [DDL - Data Definition Language](#ddl---data-definition-language)
3. [DML - Data Manipulation Language](#dml---data-manipulation-language)
4. [DQL - Data Query Language](#dql---data-query-language)
5. [JOINs](#joins)
6. [Aggregate Functions](#aggregate-functions)
7. [GROUP BY & HAVING](#group-by--having)
8. [Subqueries](#subqueries)
9. [Window Functions](#window-functions)
10. [Common Table Expressions (CTEs)](#common-table-expressions-ctes)
11. [Set Operations](#set-operations)
12. [String Functions](#string-functions)
13. [Date/Time Functions](#datetime-functions)
14. [Mathematical Functions](#mathematical-functions)
15. [CASE Expressions](#case-expressions)
16. [NULL Handling](#null-handling)
17. [Constraints](#constraints)
18. [Indexes](#indexes)
19. [Views](#views)
20. [Transactions](#transactions)
21. [Advanced Topics](#advanced-topics)
22. [Performance & Optimization](#performance--optimization)
23. [Common Interview Patterns](#common-interview-patterns)

---

## Data Types

### Numeric Types
| Type | Description | Range | Storage |
|------|-------------|-------|---------|
| `SMALLINT` | Small integer | -32,768 to 32,767 | 2 bytes |
| `INTEGER` / `INT` | Standard integer | -2,147,483,648 to 2,147,483,647 | 4 bytes |
| `BIGINT` | Large integer | -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807 | 8 bytes |
| `DECIMAL(p,s)` / `NUMERIC(p,s)` | Exact decimal | Up to 131,072 digits before decimal, 16,383 after | Variable |
| `REAL` | Single precision float | 6 decimal digits precision | 4 bytes |
| `DOUBLE PRECISION` | Double precision float | 15 decimal digits precision | 8 bytes |
| `SERIAL` | Auto-incrementing integer | Same as INTEGER | 4 bytes |
| `BIGSERIAL` | Auto-incrementing big integer | Same as BIGINT | 8 bytes |

**Note**: Use `DECIMAL/NUMERIC` for money to avoid floating-point errors!

### Character Types
| Type | Description | Max Size |
|------|-------------|----------|
| `CHAR(n)` | Fixed-length character | n characters (padded with spaces) |
| `VARCHAR(n)` | Variable-length character | n characters |
| `TEXT` | Variable unlimited length | Unlimited |

**Tip**: `VARCHAR` without limit behaves like `TEXT` in PostgreSQL

### Boolean Type
- `BOOLEAN` - Values: `TRUE`, `FALSE`, `NULL`
- Can use `'t'`, `'f'`, `'yes'`, `'no'`, `'1'`, `'0'`

### Date/Time Types
| Type | Description | Format Example |
|------|-------------|----------------|
| `DATE` | Calendar date | `'2025-12-20'` |
| `TIME` | Time of day | `'14:30:00'` |
| `TIMESTAMP` | Date and time | `'2025-12-20 14:30:00'` |
| `TIMESTAMPTZ` | Timestamp with timezone | `'2025-12-20 14:30:00+00'` |
| `INTERVAL` | Time interval | `'1 day'`, `'2 hours'` |

### Other Important Types
- `UUID` - Universally unique identifier
- `JSON` / `JSONB` - JSON data (JSONB is binary, faster)
- `ARRAY` - Array of elements: `INTEGER[]`, `TEXT[]`
- `BYTEA` - Binary data
- `ENUM` - User-defined enumeration type

---

## DDL - Data Definition Language

### CREATE TABLE

```sql
-- Basic table creation
CREATE TABLE employees (
    employee_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    hire_date DATE DEFAULT CURRENT_DATE,
    salary DECIMAL(10,2) CHECK (salary > 0),
    department_id INTEGER REFERENCES departments(department_id),
    is_active BOOLEAN DEFAULT TRUE
);

-- Create table from query
CREATE TABLE high_earners AS
SELECT * FROM employees WHERE salary > 100000;

-- Temporary table (auto-deleted at end of session)
CREATE TEMP TABLE temp_data (
    id INTEGER,
    value TEXT
);
```

### ALTER TABLE

```sql
-- Add column
ALTER TABLE employees ADD COLUMN phone VARCHAR(20);

-- Drop column
ALTER TABLE employees DROP COLUMN phone;

-- Rename column
ALTER TABLE employees RENAME COLUMN email TO email_address;

-- Change data type
ALTER TABLE employees ALTER COLUMN salary TYPE NUMERIC(12,2);

-- Set default value
ALTER TABLE employees ALTER COLUMN is_active SET DEFAULT TRUE;

-- Drop default
ALTER TABLE employees ALTER COLUMN is_active DROP DEFAULT;

-- Add constraint
ALTER TABLE employees ADD CONSTRAINT unique_email UNIQUE (email);

-- Drop constraint
ALTER TABLE employees DROP CONSTRAINT unique_email;

-- Rename table
ALTER TABLE employees RENAME TO staff;
```

### DROP TABLE

```sql
-- Drop table
DROP TABLE employees;

-- Drop if exists (no error if doesn't exist)
DROP TABLE IF EXISTS employees;

-- Drop with cascade (drops dependent objects)
DROP TABLE employees CASCADE;
```

### TRUNCATE TABLE

```sql
-- Remove all rows (faster than DELETE)
TRUNCATE TABLE employees;

-- Reset auto-increment
TRUNCATE TABLE employees RESTART IDENTITY;

-- With cascade (truncate referencing tables)
TRUNCATE TABLE employees CASCADE;
```

---

## DML - Data Manipulation Language

### INSERT

```sql
-- Insert single row
INSERT INTO employees (first_name, last_name, email, salary)
VALUES ('John', 'Doe', 'john@example.com', 75000);

-- Insert multiple rows
INSERT INTO employees (first_name, last_name, salary)
VALUES
    ('Jane', 'Smith', 80000),
    ('Bob', 'Johnson', 70000),
    ('Alice', 'Williams', 85000);

-- Insert from SELECT
INSERT INTO archive_employees
SELECT * FROM employees WHERE hire_date < '2020-01-01';

-- Insert with RETURNING (get inserted values)
INSERT INTO employees (first_name, last_name)
VALUES ('Tom', 'Brown')
RETURNING employee_id, hire_date;

-- Insert with ON CONFLICT (UPSERT)
INSERT INTO employees (employee_id, first_name, last_name, salary)
VALUES (1, 'John', 'Doe', 75000)
ON CONFLICT (employee_id)
DO UPDATE SET salary = EXCLUDED.salary;

-- Do nothing on conflict
INSERT INTO employees (email, first_name, last_name)
VALUES ('john@example.com', 'John', 'Doe')
ON CONFLICT (email) DO NOTHING;
```

### UPDATE

```sql
-- Update single column
UPDATE employees
SET salary = 80000
WHERE employee_id = 1;

-- Update multiple columns
UPDATE employees
SET
    salary = salary * 1.1,
    last_modified = CURRENT_TIMESTAMP
WHERE department_id = 5;

-- Update from another table
UPDATE employees e
SET department_name = d.name
FROM departments d
WHERE e.department_id = d.department_id;

-- Update with subquery
UPDATE employees
SET salary = (SELECT AVG(salary) FROM employees)
WHERE salary IS NULL;

-- Update with RETURNING
UPDATE employees
SET salary = salary * 1.1
WHERE employee_id = 1
RETURNING employee_id, salary;
```

### DELETE

```sql
-- Delete specific rows
DELETE FROM employees
WHERE employee_id = 1;

-- Delete with condition
DELETE FROM employees
WHERE hire_date < '2015-01-01';

-- Delete using JOIN (via subquery)
DELETE FROM employees
WHERE department_id IN (
    SELECT department_id
    FROM departments
    WHERE location = 'Remote'
);

-- Delete with RETURNING
DELETE FROM employees
WHERE employee_id = 1
RETURNING *;

-- Delete all rows (use TRUNCATE instead for better performance)
DELETE FROM employees;
```

---

## DQL - Data Query Language

### Basic SELECT

```sql
-- Select all columns
SELECT * FROM employees;

-- Select specific columns
SELECT first_name, last_name, salary FROM employees;

-- Select with alias
SELECT
    first_name AS "First Name",
    last_name AS "Last Name",
    salary * 12 AS annual_salary
FROM employees;

-- Select distinct values
SELECT DISTINCT department_id FROM employees;

-- Select distinct combinations
SELECT DISTINCT department_id, job_title FROM employees;
```

### WHERE Clause

```sql
-- Comparison operators: =, !=, <>, <, >, <=, >=
SELECT * FROM employees WHERE salary > 70000;

-- Logical operators: AND, OR, NOT
SELECT * FROM employees
WHERE salary > 70000 AND department_id = 5;

SELECT * FROM employees
WHERE department_id = 5 OR department_id = 10;

SELECT * FROM employees WHERE NOT (salary < 50000);

-- BETWEEN
SELECT * FROM employees
WHERE salary BETWEEN 50000 AND 80000;

-- IN
SELECT * FROM employees
WHERE department_id IN (5, 10, 15);

-- LIKE (pattern matching)
SELECT * FROM employees WHERE last_name LIKE 'S%';  -- Starts with S
SELECT * FROM employees WHERE email LIKE '%@gmail.com';  -- Ends with
SELECT * FROM employees WHERE first_name LIKE '_ohn';  -- _ matches single char

-- ILIKE (case-insensitive)
SELECT * FROM employees WHERE last_name ILIKE 's%';

-- IS NULL / IS NOT NULL
SELECT * FROM employees WHERE email IS NULL;
SELECT * FROM employees WHERE email IS NOT NULL;

-- ANY / ALL
SELECT * FROM employees
WHERE salary > ANY (SELECT salary FROM employees WHERE department_id = 5);

SELECT * FROM employees
WHERE salary > ALL (SELECT salary FROM employees WHERE department_id = 5);

-- EXISTS
SELECT * FROM departments d
WHERE EXISTS (
    SELECT 1 FROM employees e WHERE e.department_id = d.department_id
);
```

### ORDER BY

```sql
-- Ascending order (default)
SELECT * FROM employees ORDER BY salary;
SELECT * FROM employees ORDER BY salary ASC;

-- Descending order
SELECT * FROM employees ORDER BY salary DESC;

-- Multiple columns
SELECT * FROM employees
ORDER BY department_id ASC, salary DESC;

-- Using column position
SELECT first_name, last_name, salary
FROM employees
ORDER BY 3 DESC;  -- Order by 3rd column (salary)

-- NULL handling
SELECT * FROM employees ORDER BY email NULLS FIRST;
SELECT * FROM employees ORDER BY email NULLS LAST;
```

### LIMIT & OFFSET

```sql
-- Limit results
SELECT * FROM employees LIMIT 10;

-- Pagination (skip first 10, get next 10)
SELECT * FROM employees
ORDER BY employee_id
LIMIT 10 OFFSET 10;

-- Alternative syntax (FETCH - SQL standard)
SELECT * FROM employees
ORDER BY employee_id
OFFSET 10 ROWS
FETCH FIRST 10 ROWS ONLY;
```

---

## JOINs

### Visual JOIN Reference
```
INNER JOIN: Returns only matching rows from both tables
LEFT JOIN: All from left + matching from right (NULL if no match)
RIGHT JOIN: All from right + matching from left (NULL if no match)
FULL OUTER JOIN: All rows from both tables (NULL where no match)
CROSS JOIN: Cartesian product (all combinations)
```

### INNER JOIN

```sql
-- Standard INNER JOIN
SELECT e.first_name, e.last_name, d.department_name
FROM employees e
INNER JOIN departments d ON e.department_id = d.department_id;

-- Multiple joins
SELECT e.first_name, d.department_name, l.city
FROM employees e
INNER JOIN departments d ON e.department_id = d.department_id
INNER JOIN locations l ON d.location_id = l.location_id;

-- Join with additional conditions
SELECT e.first_name, d.department_name
FROM employees e
INNER JOIN departments d
    ON e.department_id = d.department_id
    AND d.budget > 100000;

-- Self join (compare rows within same table)
SELECT
    e1.first_name AS employee,
    e2.first_name AS manager
FROM employees e1
INNER JOIN employees e2 ON e1.manager_id = e2.employee_id;
```

### LEFT JOIN (LEFT OUTER JOIN)

```sql
-- Get all employees, even without departments
SELECT e.first_name, e.last_name, d.department_name
FROM employees e
LEFT JOIN departments d ON e.department_id = d.department_id;

-- Find employees without departments
SELECT e.first_name, e.last_name
FROM employees e
LEFT JOIN departments d ON e.department_id = d.department_id
WHERE d.department_id IS NULL;
```

### RIGHT JOIN (RIGHT OUTER JOIN)

```sql
-- Get all departments, even without employees
SELECT e.first_name, d.department_name
FROM employees e
RIGHT JOIN departments d ON e.department_id = d.department_id;

-- Find departments with no employees
SELECT d.department_name
FROM employees e
RIGHT JOIN departments d ON e.department_id = d.department_id
WHERE e.employee_id IS NULL;
```

### FULL OUTER JOIN

```sql
-- Get all employees and all departments
SELECT e.first_name, d.department_name
FROM employees e
FULL OUTER JOIN departments d ON e.department_id = d.department_id;

-- Find unmatched records from both tables
SELECT e.first_name, d.department_name
FROM employees e
FULL OUTER JOIN departments d ON e.department_id = d.department_id
WHERE e.employee_id IS NULL OR d.department_id IS NULL;
```

### CROSS JOIN

```sql
-- Cartesian product (every combination)
SELECT e.first_name, d.department_name
FROM employees e
CROSS JOIN departments d;

-- Alternative syntax
SELECT e.first_name, d.department_name
FROM employees e, departments d;
```

### NATURAL JOIN (avoid in production!)

```sql
-- Joins on all columns with same name
SELECT * FROM employees NATURAL JOIN departments;
-- Dangerous: implicit join conditions can cause unexpected results
```

---

## Aggregate Functions

### Common Aggregates

```sql
-- COUNT - count rows
SELECT COUNT(*) FROM employees;  -- All rows including NULL
SELECT COUNT(email) FROM employees;  -- Non-NULL values only
SELECT COUNT(DISTINCT department_id) FROM employees;  -- Unique values

-- SUM - sum of values
SELECT SUM(salary) FROM employees;

-- AVG - average
SELECT AVG(salary) FROM employees;
SELECT AVG(salary)::NUMERIC(10,2) FROM employees;  -- Round result

-- MIN / MAX
SELECT MIN(salary), MAX(salary) FROM employees;
SELECT MIN(hire_date) FROM employees;

-- Multiple aggregates
SELECT
    COUNT(*) AS total_employees,
    AVG(salary) AS avg_salary,
    MIN(salary) AS min_salary,
    MAX(salary) AS max_salary,
    SUM(salary) AS total_payroll
FROM employees;

-- String aggregation (PostgreSQL specific)
SELECT STRING_AGG(first_name, ', ') AS all_names
FROM employees;

SELECT STRING_AGG(first_name, ', ' ORDER BY first_name) AS sorted_names
FROM employees;

-- Array aggregation
SELECT ARRAY_AGG(first_name ORDER BY first_name) AS names_array
FROM employees;

-- Statistical functions
SELECT
    STDDEV(salary) AS standard_deviation,
    VARIANCE(salary) AS variance
FROM employees;
```

---

## GROUP BY & HAVING

### GROUP BY

```sql
-- Group by single column
SELECT department_id, COUNT(*) AS employee_count
FROM employees
GROUP BY department_id;

-- Group by multiple columns
SELECT department_id, job_title, AVG(salary) AS avg_salary
FROM employees
GROUP BY department_id, job_title;

-- Group with expressions
SELECT
    EXTRACT(YEAR FROM hire_date) AS hire_year,
    COUNT(*) AS hires
FROM employees
GROUP BY EXTRACT(YEAR FROM hire_date);

-- Using column alias (PostgreSQL extension)
SELECT
    EXTRACT(YEAR FROM hire_date) AS hire_year,
    COUNT(*) AS hires
FROM employees
GROUP BY hire_year;
```

### HAVING

```sql
-- Filter grouped results
SELECT department_id, AVG(salary) AS avg_salary
FROM employees
GROUP BY department_id
HAVING AVG(salary) > 70000;

-- Multiple conditions
SELECT department_id, COUNT(*) AS employee_count
FROM employees
GROUP BY department_id
HAVING COUNT(*) > 5 AND AVG(salary) > 60000;

-- WHERE vs HAVING
SELECT department_id, AVG(salary) AS avg_salary
FROM employees
WHERE hire_date > '2020-01-01'  -- Filter BEFORE grouping
GROUP BY department_id
HAVING AVG(salary) > 70000;  -- Filter AFTER grouping
```

### GROUPING SETS, ROLLUP, CUBE

```sql
-- GROUPING SETS - multiple groupings in one query
SELECT department_id, job_title, SUM(salary)
FROM employees
GROUP BY GROUPING SETS (
    (department_id, job_title),
    (department_id),
    (job_title),
    ()  -- Grand total
);

-- ROLLUP - hierarchical aggregations
SELECT department_id, job_title, SUM(salary)
FROM employees
GROUP BY ROLLUP (department_id, job_title);
-- Produces: (dept, job), (dept), ()

-- CUBE - all combinations
SELECT department_id, job_title, SUM(salary)
FROM employees
GROUP BY CUBE (department_id, job_title);
-- Produces: (dept, job), (dept), (job), ()
```

---

## Subqueries

### Scalar Subquery (returns single value)

```sql
-- In SELECT clause
SELECT
    first_name,
    salary,
    (SELECT AVG(salary) FROM employees) AS avg_salary,
    salary - (SELECT AVG(salary) FROM employees) AS difference
FROM employees;

-- In WHERE clause
SELECT first_name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);
```

### Row Subquery

```sql
-- Compare multiple columns
SELECT * FROM employees
WHERE (department_id, salary) = (
    SELECT department_id, MAX(salary)
    FROM employees
    WHERE department_id = 5
);
```

### Table Subquery

```sql
-- In FROM clause (derived table)
SELECT dept_id, avg_sal
FROM (
    SELECT department_id AS dept_id, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY department_id
) AS dept_averages
WHERE avg_sal > 70000;

-- IN operator
SELECT first_name, last_name
FROM employees
WHERE department_id IN (
    SELECT department_id FROM departments WHERE location = 'New York'
);

-- NOT IN
SELECT first_name, last_name
FROM employees
WHERE department_id NOT IN (
    SELECT department_id FROM departments WHERE budget > 500000
);

-- ANY / SOME
SELECT first_name, salary
FROM employees
WHERE salary > ANY (
    SELECT salary FROM employees WHERE department_id = 5
);

-- ALL
SELECT first_name, salary
FROM employees
WHERE salary > ALL (
    SELECT salary FROM employees WHERE department_id = 5
);
```

### Correlated Subquery

```sql
-- Subquery references outer query
SELECT e1.first_name, e1.salary, e1.department_id
FROM employees e1
WHERE salary > (
    SELECT AVG(salary)
    FROM employees e2
    WHERE e2.department_id = e1.department_id
);

-- Find employees earning more than their department average
SELECT
    e.first_name,
    e.salary,
    (SELECT AVG(salary)
     FROM employees e2
     WHERE e2.department_id = e.department_id) AS dept_avg
FROM employees e;
```

### EXISTS / NOT EXISTS

```sql
-- Check existence
SELECT d.department_name
FROM departments d
WHERE EXISTS (
    SELECT 1 FROM employees e WHERE e.department_id = d.department_id
);

-- Find departments with no employees
SELECT d.department_name
FROM departments d
WHERE NOT EXISTS (
    SELECT 1 FROM employees e WHERE e.department_id = d.department_id
);
```

---

## Window Functions

**Syntax**: `function() OVER ([PARTITION BY ...] [ORDER BY ...] [frame_clause])`

### Ranking Functions

```sql
-- ROW_NUMBER - unique sequential number
SELECT
    first_name,
    salary,
    ROW_NUMBER() OVER (ORDER BY salary DESC) AS row_num
FROM employees;

-- RANK - same rank for ties, gaps in sequence
SELECT
    first_name,
    salary,
    RANK() OVER (ORDER BY salary DESC) AS rank
FROM employees;
-- If two people have rank 2, next rank is 4

-- DENSE_RANK - same rank for ties, no gaps
SELECT
    first_name,
    salary,
    DENSE_RANK() OVER (ORDER BY salary DESC) AS dense_rank
FROM employees;
-- If two people have rank 2, next rank is 3

-- NTILE - divide into N buckets
SELECT
    first_name,
    salary,
    NTILE(4) OVER (ORDER BY salary DESC) AS quartile
FROM employees;
```

### Partition By (separate rankings per group)

```sql
-- Rank within each department
SELECT
    department_id,
    first_name,
    salary,
    RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS dept_rank
FROM employees;

-- Row number within partition
SELECT
    department_id,
    first_name,
    hire_date,
    ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY hire_date) AS hire_order
FROM employees;
```

### Aggregate Window Functions

```sql
-- Running total
SELECT
    hire_date,
    salary,
    SUM(salary) OVER (ORDER BY hire_date) AS running_total
FROM employees;

-- Average within partition
SELECT
    department_id,
    first_name,
    salary,
    AVG(salary) OVER (PARTITION BY department_id) AS dept_avg
FROM employees;

-- Count
SELECT
    department_id,
    first_name,
    COUNT(*) OVER (PARTITION BY department_id) AS dept_count
FROM employees;
```

### Value Functions

```sql
-- LEAD - next row value
SELECT
    employee_id,
    salary,
    LEAD(salary) OVER (ORDER BY employee_id) AS next_salary
FROM employees;

-- LAG - previous row value
SELECT
    employee_id,
    salary,
    LAG(salary) OVER (ORDER BY employee_id) AS prev_salary,
    salary - LAG(salary) OVER (ORDER BY employee_id) AS salary_diff
FROM employees;

-- FIRST_VALUE
SELECT
    department_id,
    first_name,
    salary,
    FIRST_VALUE(salary) OVER (PARTITION BY department_id ORDER BY salary DESC) AS highest_salary
FROM employees;

-- LAST_VALUE (need frame clause!)
SELECT
    department_id,
    first_name,
    salary,
    LAST_VALUE(salary) OVER (
        PARTITION BY department_id
        ORDER BY salary DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS lowest_salary
FROM employees;

-- NTH_VALUE
SELECT
    department_id,
    first_name,
    salary,
    NTH_VALUE(salary, 2) OVER (
        PARTITION BY department_id
        ORDER BY salary DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS second_highest
FROM employees;
```

### Frame Clauses

```sql
-- ROWS BETWEEN
SELECT
    hire_date,
    salary,
    AVG(salary) OVER (
        ORDER BY hire_date
        ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING
    ) AS moving_avg_5
FROM employees;

-- RANGE BETWEEN (value-based)
SELECT
    hire_date,
    salary,
    SUM(salary) OVER (
        ORDER BY salary
        RANGE BETWEEN 10000 PRECEDING AND 10000 FOLLOWING
    ) AS salary_range_sum
FROM employees;

-- Frame specifications:
-- ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW (default)
-- ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING
-- ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
-- RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
```

---

## Common Table Expressions (CTEs)

### Basic CTE

```sql
-- Single CTE
WITH high_earners AS (
    SELECT * FROM employees WHERE salary > 80000
)
SELECT first_name, last_name, salary
FROM high_earners
ORDER BY salary DESC;

-- Multiple CTEs
WITH
dept_avg AS (
    SELECT department_id, AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department_id
),
high_depts AS (
    SELECT department_id
    FROM dept_avg
    WHERE avg_salary > 70000
)
SELECT e.first_name, e.salary
FROM employees e
JOIN high_depts h ON e.department_id = h.department_id;
```

### Recursive CTE

```sql
-- Employee hierarchy (org chart)
WITH RECURSIVE org_chart AS (
    -- Anchor: top-level employees (no manager)
    SELECT employee_id, first_name, manager_id, 1 AS level
    FROM employees
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive: employees reporting to previous level
    SELECT e.employee_id, e.first_name, e.manager_id, oc.level + 1
    FROM employees e
    JOIN org_chart oc ON e.manager_id = oc.employee_id
)
SELECT * FROM org_chart ORDER BY level, first_name;

-- Generate series (numbers)
WITH RECURSIVE numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n < 100
)
SELECT * FROM numbers;

-- Date series
WITH RECURSIVE date_series AS (
    SELECT '2025-01-01'::DATE AS date
    UNION ALL
    SELECT date + INTERVAL '1 day'
    FROM date_series
    WHERE date < '2025-12-31'::DATE
)
SELECT * FROM date_series;
```

### CTE vs Subquery

```sql
-- Readability: CTE is often clearer
-- Reusability: CTE can be referenced multiple times

-- CTE example
WITH dept_stats AS (
    SELECT department_id, AVG(salary) AS avg_sal, COUNT(*) AS emp_count
    FROM employees
    GROUP BY department_id
)
SELECT * FROM dept_stats WHERE avg_sal > 70000
UNION ALL
SELECT * FROM dept_stats WHERE emp_count > 10;

-- Performance: Generally similar, but CTE can be materialized
-- Use MATERIALIZED hint to force materialization
WITH dept_stats AS MATERIALIZED (
    SELECT department_id, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY department_id
)
SELECT * FROM dept_stats;
```

---

## Set Operations

### UNION

```sql
-- Combine results, remove duplicates
SELECT first_name, last_name FROM employees
UNION
SELECT first_name, last_name FROM contractors;

-- UNION ALL - keep duplicates (faster)
SELECT first_name FROM employees
UNION ALL
SELECT first_name FROM contractors;
```

### INTERSECT

```sql
-- Return only rows in both queries
SELECT email FROM employees
INTERSECT
SELECT email FROM newsletter_subscribers;
```

### EXCEPT

```sql
-- Return rows in first query but not in second
SELECT email FROM employees
EXCEPT
SELECT email FROM newsletter_subscribers;

-- Find employees not in contractors
SELECT first_name, last_name FROM employees
EXCEPT
SELECT first_name, last_name FROM contractors;
```

**Rules for Set Operations:**
- Must have same number of columns
- Corresponding columns must have compatible data types
- Column names come from first query
- ORDER BY applies to final result only

---

## String Functions

```sql
-- Concatenation
SELECT first_name || ' ' || last_name AS full_name FROM employees;
SELECT CONCAT(first_name, ' ', last_name) AS full_name FROM employees;
SELECT CONCAT_WS(', ', last_name, first_name) AS name FROM employees;  -- With separator

-- Case conversion
SELECT UPPER(first_name) FROM employees;
SELECT LOWER(email) FROM employees;
SELECT INITCAP('hello world');  -- 'Hello World'

-- Substring
SELECT SUBSTRING('PostgreSQL' FROM 1 FOR 8);  -- 'Postgres'
SELECT SUBSTRING(email FROM POSITION('@' IN email) + 1) AS domain FROM employees;
SELECT LEFT('PostgreSQL', 8);  -- 'Postgres'
SELECT RIGHT('PostgreSQL', 3);  -- 'SQL'

-- Length
SELECT LENGTH('PostgreSQL');  -- 10
SELECT CHAR_LENGTH('PostgreSQL');  -- 10 (character count)

-- Trimming
SELECT TRIM('  hello  ');  -- 'hello'
SELECT LTRIM('  hello');  -- 'hello'
SELECT RTRIM('hello  ');  -- 'hello'
SELECT TRIM(BOTH 'x' FROM 'xxxhelloxxx');  -- 'hello'

-- Padding
SELECT LPAD('42', 5, '0');  -- '00042'
SELECT RPAD('hi', 5, '*');  -- 'hi***'

-- Position/Index
SELECT POSITION('SQL' IN 'PostgreSQL');  -- 9
SELECT STRPOS('PostgreSQL', 'SQL');  -- 9 (same as POSITION)

-- Replace
SELECT REPLACE('PostgreSQL', 'SQL', 'Database');  -- 'PostgreDatabase'

-- Repeat
SELECT REPEAT('*', 5);  -- '*****'

-- Reverse
SELECT REVERSE('PostgreSQL');  -- 'LQSertsgoP'

-- Split
SELECT SPLIT_PART('a,b,c', ',', 2);  -- 'b'
SELECT STRING_TO_ARRAY('a,b,c', ',');  -- {a,b,c}

-- Regular expressions
SELECT 'PostgreSQL' ~ 'Post';  -- true (case-sensitive match)
SELECT 'PostgreSQL' ~* 'post';  -- true (case-insensitive)
SELECT REGEXP_REPLACE('PostgreSQL 14', '[0-9]+', '15');  -- 'PostgreSQL 15'
SELECT REGEXP_MATCHES('abc123def456', '[0-9]+', 'g');  -- {123}, {456}
```

---

## Date/Time Functions

### Current Date/Time

```sql
-- Current values
SELECT CURRENT_DATE;  -- 2025-12-20
SELECT CURRENT_TIME;  -- 14:30:00.123456+00
SELECT CURRENT_TIMESTAMP;  -- 2025-12-20 14:30:00.123456+00
SELECT NOW();  -- Same as CURRENT_TIMESTAMP
SELECT LOCALTIMESTAMP;  -- Without timezone
SELECT CLOCK_TIMESTAMP();  -- Actual current time (changes during query)
```

### Extracting Parts

```sql
-- EXTRACT
SELECT EXTRACT(YEAR FROM hire_date) FROM employees;
SELECT EXTRACT(MONTH FROM hire_date) FROM employees;
SELECT EXTRACT(DAY FROM hire_date) FROM employees;
SELECT EXTRACT(DOW FROM hire_date) FROM employees;  -- Day of week (0=Sunday)
SELECT EXTRACT(DOY FROM hire_date) FROM employees;  -- Day of year (1-366)
SELECT EXTRACT(WEEK FROM hire_date) FROM employees;  -- Week number
SELECT EXTRACT(QUARTER FROM hire_date) FROM employees;  -- Quarter (1-4)
SELECT EXTRACT(EPOCH FROM hire_date) FROM employees;  -- Unix timestamp

-- DATE_PART (same as EXTRACT)
SELECT DATE_PART('year', hire_date) FROM employees;

-- Specific functions
SELECT DATE_TRUNC('month', hire_date) FROM employees;  -- First day of month
SELECT DATE_TRUNC('year', hire_date) FROM employees;  -- First day of year
SELECT DATE_TRUNC('week', hire_date) FROM employees;  -- First day of week
```

### Date Arithmetic

```sql
-- Add/subtract intervals
SELECT hire_date + INTERVAL '1 day' FROM employees;
SELECT hire_date + INTERVAL '1 month' FROM employees;
SELECT hire_date + INTERVAL '1 year' FROM employees;
SELECT hire_date - INTERVAL '30 days' FROM employees;
SELECT NOW() - INTERVAL '1 hour';

-- Date difference
SELECT AGE(CURRENT_DATE, hire_date) FROM employees;  -- interval
SELECT CURRENT_DATE - hire_date AS days_employed FROM employees;  -- integer days

-- Generate date series
SELECT generate_series(
    '2025-01-01'::DATE,
    '2025-12-31'::DATE,
    INTERVAL '1 day'
) AS date;
```

### Formatting

```sql
-- TO_CHAR (date to string)
SELECT TO_CHAR(hire_date, 'YYYY-MM-DD') FROM employees;
SELECT TO_CHAR(hire_date, 'Month DD, YYYY') FROM employees;
SELECT TO_CHAR(hire_date, 'Day') FROM employees;  -- 'Monday   '
SELECT TO_CHAR(hire_date, 'Dy') FROM employees;  -- 'Mon'
SELECT TO_CHAR(hire_date, 'FMMonth DD, YYYY') FROM employees;  -- No padding

-- TO_DATE (string to date)
SELECT TO_DATE('2025-12-20', 'YYYY-MM-DD');
SELECT TO_DATE('20/12/2025', 'DD/MM/YYYY');

-- TO_TIMESTAMP
SELECT TO_TIMESTAMP('2025-12-20 14:30:00', 'YYYY-MM-DD HH24:MI:SS');

-- Common format patterns:
-- YYYY - 4-digit year
-- MM - 2-digit month
-- DD - 2-digit day
-- HH24 - 24-hour format
-- HH - 12-hour format
-- MI - minutes
-- SS - seconds
-- Day/Dy - day name
-- Month/Mon - month name
```

---

## Mathematical Functions

```sql
-- Basic arithmetic
SELECT 10 + 5;  -- 15
SELECT 10 - 5;  -- 5
SELECT 10 * 5;  -- 50
SELECT 10 / 5;  -- 2
SELECT 10 % 3;  -- 1 (modulo)
SELECT 10 ^ 2;  -- 100 (power)

-- Rounding
SELECT ROUND(42.4567, 2);  -- 42.46
SELECT CEIL(42.1);  -- 43
SELECT CEILING(42.1);  -- 43
SELECT FLOOR(42.9);  -- 42
SELECT TRUNC(42.4567, 2);  -- 42.45 (truncate without rounding)

-- Absolute value
SELECT ABS(-42);  -- 42

-- Power and roots
SELECT POWER(2, 10);  -- 1024
SELECT SQRT(16);  -- 4
SELECT CBRT(27);  -- 3 (cube root)

-- Exponential and logarithms
SELECT EXP(1);  -- e (2.718...)
SELECT LN(2.718281828);  -- 1 (natural log)
SELECT LOG(100);  -- 2 (base 10 log)
SELECT LOG(2, 8);  -- 3 (log base 2 of 8)

-- Trigonometric
SELECT PI();  -- 3.14159...
SELECT SIN(PI()/2);  -- 1
SELECT COS(0);  -- 1
SELECT TAN(PI()/4);  -- 1

-- Random
SELECT RANDOM();  -- Random between 0 and 1
SELECT RANDOM() * 100;  -- Random between 0 and 100
SELECT FLOOR(RANDOM() * 100 + 1)::INT;  -- Random integer 1-100

-- Sign
SELECT SIGN(-42);  -- -1
SELECT SIGN(42);  -- 1
SELECT SIGN(0);  -- 0

-- Greatest/Least
SELECT GREATEST(10, 20, 5, 30);  -- 30
SELECT LEAST(10, 20, 5, 30);  -- 5
```

---

## CASE Expressions

### Simple CASE

```sql
-- Simple CASE (equality check)
SELECT
    first_name,
    department_id,
    CASE department_id
        WHEN 1 THEN 'Engineering'
        WHEN 2 THEN 'Sales'
        WHEN 3 THEN 'Marketing'
        ELSE 'Other'
    END AS department_name
FROM employees;
```

### Searched CASE

```sql
-- Searched CASE (complex conditions)
SELECT
    first_name,
    salary,
    CASE
        WHEN salary < 50000 THEN 'Low'
        WHEN salary BETWEEN 50000 AND 80000 THEN 'Medium'
        WHEN salary > 80000 THEN 'High'
        ELSE 'Unknown'
    END AS salary_level
FROM employees;

-- Multiple conditions
SELECT
    first_name,
    CASE
        WHEN salary > 100000 AND department_id = 1 THEN 'Senior Engineer'
        WHEN salary > 80000 AND department_id = 1 THEN 'Engineer'
        WHEN salary > 100000 THEN 'Senior Staff'
        ELSE 'Staff'
    END AS job_level
FROM employees;
```

### CASE in Aggregate Functions

```sql
-- Conditional counting
SELECT
    COUNT(*) AS total,
    COUNT(CASE WHEN salary > 80000 THEN 1 END) AS high_earners,
    COUNT(CASE WHEN salary <= 80000 THEN 1 END) AS regular_earners
FROM employees;

-- Conditional summing
SELECT
    SUM(CASE WHEN department_id = 1 THEN salary ELSE 0 END) AS eng_payroll,
    SUM(CASE WHEN department_id = 2 THEN salary ELSE 0 END) AS sales_payroll
FROM employees;

-- Pivot table simulation
SELECT
    EXTRACT(YEAR FROM hire_date) AS year,
    COUNT(CASE WHEN department_id = 1 THEN 1 END) AS eng_hires,
    COUNT(CASE WHEN department_id = 2 THEN 1 END) AS sales_hires,
    COUNT(CASE WHEN department_id = 3 THEN 1 END) AS marketing_hires
FROM employees
GROUP BY EXTRACT(YEAR FROM hire_date);
```

### CASE in ORDER BY

```sql
-- Custom sorting
SELECT first_name, department_id
FROM employees
ORDER BY
    CASE department_id
        WHEN 1 THEN 1
        WHEN 3 THEN 2
        WHEN 2 THEN 3
        ELSE 4
    END;
```

---

## NULL Handling

### NULL Checks

```sql
-- IS NULL / IS NOT NULL
SELECT * FROM employees WHERE email IS NULL;
SELECT * FROM employees WHERE email IS NOT NULL;

-- NULLIF - return NULL if values are equal
SELECT NULLIF(department_id, 0) FROM employees;  -- NULL if dept_id is 0

-- COALESCE - return first non-NULL value
SELECT COALESCE(email, phone, 'No contact') FROM employees;
SELECT COALESCE(middle_name, '') FROM employees;  -- Empty string if NULL
```

### NULL in Comparisons

```sql
-- NULL in arithmetic (result is NULL)
SELECT salary + NULL FROM employees;  -- Returns NULL

-- NULL in comparisons (result is NULL, not TRUE or FALSE)
SELECT * FROM employees WHERE salary = NULL;  -- Wrong! Returns nothing
SELECT * FROM employees WHERE salary IS NULL;  -- Correct

-- NULL in logic
SELECT * FROM employees WHERE salary > 50000 OR salary IS NULL;
SELECT * FROM employees WHERE NOT (salary <= 50000) AND salary IS NOT NULL;
```

### NULL in Aggregate Functions

```sql
-- Aggregates ignore NULLs (except COUNT(*))
SELECT COUNT(*) FROM employees;  -- Counts all rows
SELECT COUNT(email) FROM employees;  -- Counts non-NULL emails
SELECT AVG(salary) FROM employees;  -- Average of non-NULL salaries
SELECT SUM(bonus) FROM employees;  -- Sum of non-NULL bonuses
```

### NULL-Safe Comparisons

```sql
-- IS DISTINCT FROM (treats NULL as a value)
SELECT * FROM employees WHERE email IS DISTINCT FROM 'john@example.com';
-- Returns TRUE if email is NULL or different from 'john@example.com'

SELECT * FROM employees WHERE email IS NOT DISTINCT FROM NULL;
-- Same as: WHERE email IS NULL
```

---

## Constraints

### PRIMARY KEY

```sql
-- Single column
CREATE TABLE employees (
    employee_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50)
);

-- Composite primary key
CREATE TABLE order_items (
    order_id INTEGER,
    product_id INTEGER,
    quantity INTEGER,
    PRIMARY KEY (order_id, product_id)
);

-- Add after creation
ALTER TABLE employees ADD PRIMARY KEY (employee_id);
```

### FOREIGN KEY

```sql
-- On creation
CREATE TABLE employees (
    employee_id SERIAL PRIMARY KEY,
    department_id INTEGER REFERENCES departments(department_id)
);

-- With named constraint
CREATE TABLE employees (
    employee_id SERIAL PRIMARY KEY,
    department_id INTEGER,
    CONSTRAINT fk_department FOREIGN KEY (department_id)
        REFERENCES departments(department_id)
);

-- With referential actions
CREATE TABLE employees (
    employee_id SERIAL PRIMARY KEY,
    department_id INTEGER REFERENCES departments(department_id)
        ON DELETE CASCADE          -- Delete employee if department deleted
        ON UPDATE CASCADE          -- Update employee if department_id changes
);

-- Other actions:
-- ON DELETE SET NULL - Set to NULL if parent deleted
-- ON DELETE SET DEFAULT - Set to default if parent deleted
-- ON DELETE RESTRICT - Prevent deletion of parent (default)
-- ON DELETE NO ACTION - Same as RESTRICT

-- Add after creation
ALTER TABLE employees
ADD CONSTRAINT fk_department
FOREIGN KEY (department_id) REFERENCES departments(department_id);
```

### UNIQUE

```sql
-- Single column
CREATE TABLE employees (
    employee_id SERIAL PRIMARY KEY,
    email VARCHAR(100) UNIQUE
);

-- Multiple columns (combination must be unique)
CREATE TABLE employees (
    employee_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    department_id INTEGER,
    UNIQUE (first_name, last_name, department_id)
);

-- Add after creation
ALTER TABLE employees ADD UNIQUE (email);
ALTER TABLE employees ADD CONSTRAINT unique_email UNIQUE (email);
```

### NOT NULL

```sql
-- On creation
CREATE TABLE employees (
    employee_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL
);

-- Add after creation
ALTER TABLE employees ALTER COLUMN email SET NOT NULL;

-- Remove
ALTER TABLE employees ALTER COLUMN email DROP NOT NULL;
```

### CHECK

```sql
-- Simple check
CREATE TABLE employees (
    employee_id SERIAL PRIMARY KEY,
    salary DECIMAL(10,2) CHECK (salary > 0),
    age INTEGER CHECK (age >= 18 AND age <= 65)
);

-- Named constraint
CREATE TABLE employees (
    employee_id SERIAL PRIMARY KEY,
    salary DECIMAL(10,2),
    CONSTRAINT positive_salary CHECK (salary > 0)
);

-- Multiple columns
CREATE TABLE employees (
    employee_id SERIAL PRIMARY KEY,
    start_date DATE,
    end_date DATE,
    CHECK (end_date > start_date OR end_date IS NULL)
);

-- Add after creation
ALTER TABLE employees ADD CHECK (salary > 0);
ALTER TABLE employees ADD CONSTRAINT positive_salary CHECK (salary > 0);
```

### DEFAULT

```sql
-- Set default value
CREATE TABLE employees (
    employee_id SERIAL PRIMARY KEY,
    hire_date DATE DEFAULT CURRENT_DATE,
    is_active BOOLEAN DEFAULT TRUE,
    status VARCHAR(20) DEFAULT 'active'
);

-- Add after creation
ALTER TABLE employees ALTER COLUMN is_active SET DEFAULT TRUE;

-- Remove default
ALTER TABLE employees ALTER COLUMN is_active DROP DEFAULT;
```

---

## Indexes

### Create Index

```sql
-- Simple index
CREATE INDEX idx_employees_last_name ON employees(last_name);

-- Unique index (enforces uniqueness)
CREATE UNIQUE INDEX idx_employees_email ON employees(email);

-- Multi-column index
CREATE INDEX idx_employees_name ON employees(last_name, first_name);

-- Conditional/Partial index
CREATE INDEX idx_active_employees
ON employees(last_name)
WHERE is_active = TRUE;

-- Expression index
CREATE INDEX idx_lower_email ON employees(LOWER(email));

-- Concurrent creation (doesn't lock table)
CREATE INDEX CONCURRENTLY idx_employees_dept ON employees(department_id);
```

### Index Types

```sql
-- B-tree (default, best for most cases)
CREATE INDEX idx_employees_salary ON employees USING BTREE (salary);

-- Hash (only for equality comparisons)
CREATE INDEX idx_employees_email ON employees USING HASH (email);

-- GIN (for arrays, JSONB, full-text search)
CREATE INDEX idx_employees_skills ON employees USING GIN (skills);

-- GiST (for geometric data, full-text search)
CREATE INDEX idx_locations_point ON locations USING GIST (coordinates);

-- BRIN (for very large tables with natural ordering)
CREATE INDEX idx_logs_created_at ON logs USING BRIN (created_at);
```

### Manage Indexes

```sql
-- Drop index
DROP INDEX idx_employees_last_name;

-- Drop if exists
DROP INDEX IF EXISTS idx_employees_last_name;

-- Drop concurrently
DROP INDEX CONCURRENTLY idx_employees_last_name;

-- Rebuild index
REINDEX INDEX idx_employees_last_name;
REINDEX TABLE employees;

-- List indexes
SELECT * FROM pg_indexes WHERE tablename = 'employees';
```

### When to Use Indexes

**Create indexes when:**
- Column frequently used in WHERE clauses
- Column used in JOIN conditions
- Column used in ORDER BY
- Foreign key columns
- Columns used in GROUP BY

**Avoid indexes when:**
- Table is very small
- Column has low cardinality (few unique values)
- Frequent INSERT/UPDATE/DELETE operations
- Column rarely queried

---

## Views

### Create View

```sql
-- Simple view
CREATE VIEW active_employees AS
SELECT employee_id, first_name, last_name, salary
FROM employees
WHERE is_active = TRUE;

-- Complex view with joins
CREATE VIEW employee_details AS
SELECT
    e.employee_id,
    e.first_name,
    e.last_name,
    d.department_name,
    l.city
FROM employees e
JOIN departments d ON e.department_id = d.department_id
JOIN locations l ON d.location_id = l.location_id;

-- View with aggregations
CREATE VIEW department_stats AS
SELECT
    department_id,
    COUNT(*) AS employee_count,
    AVG(salary) AS avg_salary,
    MAX(salary) AS max_salary
FROM employees
GROUP BY department_id;
```

### Materialized Views

```sql
-- Create materialized view (data is stored)
CREATE MATERIALIZED VIEW dept_summary AS
SELECT
    department_id,
    COUNT(*) AS employee_count,
    AVG(salary) AS avg_salary
FROM employees
GROUP BY department_id;

-- Query like a table
SELECT * FROM dept_summary;

-- Refresh data
REFRESH MATERIALIZED VIEW dept_summary;

-- Refresh concurrently (non-blocking, requires unique index)
CREATE UNIQUE INDEX idx_dept_summary ON dept_summary(department_id);
REFRESH MATERIALIZED VIEW CONCURRENTLY dept_summary;
```

### Manage Views

```sql
-- Replace view
CREATE OR REPLACE VIEW active_employees AS
SELECT employee_id, first_name, last_name, email, salary
FROM employees
WHERE is_active = TRUE;

-- Drop view
DROP VIEW active_employees;
DROP VIEW IF EXISTS active_employees;

-- Drop with cascade (drop dependent views)
DROP VIEW active_employees CASCADE;

-- Drop materialized view
DROP MATERIALIZED VIEW dept_summary;
```

---

## Transactions

### Basic Transaction

```sql
-- Start transaction
BEGIN;
-- or
START TRANSACTION;

-- Execute queries
UPDATE accounts SET balance = balance - 100 WHERE account_id = 1;
UPDATE accounts SET balance = balance + 100 WHERE account_id = 2;

-- Commit (make changes permanent)
COMMIT;

-- Or rollback (undo changes)
ROLLBACK;
```

### Savepoints

```sql
BEGIN;

UPDATE employees SET salary = salary * 1.1 WHERE department_id = 1;

SAVEPOINT dept1_done;

UPDATE employees SET salary = salary * 1.1 WHERE department_id = 2;

-- Oops, rollback to savepoint
ROLLBACK TO SAVEPOINT dept1_done;

-- Continue with different operation
UPDATE employees SET salary = salary * 1.05 WHERE department_id = 2;

COMMIT;
```

### Transaction Isolation Levels

```sql
-- Read Uncommitted (not supported in PostgreSQL, treated as Read Committed)
-- Read Committed (default)
BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED;

-- Repeatable Read
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;

-- Serializable (strictest)
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;

-- Set for session
SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL REPEATABLE READ;
```

### ACID Properties

- **Atomicity**: All or nothing (COMMIT/ROLLBACK)
- **Consistency**: Database goes from one valid state to another
- **Isolation**: Concurrent transactions don't interfere
- **Durability**: Committed changes persist

---

## Advanced Topics

### LATERAL Joins

```sql
-- LATERAL allows subquery to reference earlier tables
SELECT
    d.department_name,
    top_earner.first_name,
    top_earner.salary
FROM departments d
CROSS JOIN LATERAL (
    SELECT first_name, salary
    FROM employees e
    WHERE e.department_id = d.department_id
    ORDER BY salary DESC
    LIMIT 1
) AS top_earner;
```

### FILTER Clause (Aggregate Filtering)

```sql
-- Conditional aggregation (cleaner than CASE)
SELECT
    department_id,
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE salary > 80000) AS high_earners,
    AVG(salary) FILTER (WHERE hire_date > '2020-01-01') AS avg_recent_salary
FROM employees
GROUP BY department_id;
```

### DISTINCT ON

```sql
-- Get first row per group (PostgreSQL specific)
SELECT DISTINCT ON (department_id)
    department_id,
    first_name,
    salary
FROM employees
ORDER BY department_id, salary DESC;
-- Returns highest paid employee per department
```

### RETURNING Clause

```sql
-- Get values from INSERT/UPDATE/DELETE
INSERT INTO employees (first_name, last_name)
VALUES ('John', 'Doe')
RETURNING employee_id, hire_date;

UPDATE employees
SET salary = salary * 1.1
WHERE department_id = 5
RETURNING employee_id, first_name, salary;

DELETE FROM employees
WHERE is_active = FALSE
RETURNING employee_id, first_name;
```

### Array Operations

```sql
-- Array creation
SELECT ARRAY[1, 2, 3, 4, 5];
SELECT ARRAY['a', 'b', 'c'];

-- Array from query
SELECT ARRAY(SELECT first_name FROM employees);

-- Array access (1-indexed!)
SELECT (ARRAY[10, 20, 30])[2];  -- 20

-- Array functions
SELECT ARRAY_LENGTH(ARRAY[1,2,3], 1);  -- 3
SELECT ARRAY_APPEND(ARRAY[1,2], 3);  -- {1,2,3}
SELECT ARRAY_PREPEND(0, ARRAY[1,2]);  -- {0,1,2}
SELECT ARRAY_CAT(ARRAY[1,2], ARRAY[3,4]);  -- {1,2,3,4}
SELECT UNNEST(ARRAY[1,2,3]);  -- Expands to rows: 1, 2, 3

-- Array operators
SELECT ARRAY[1,2,3] || 4;  -- {1,2,3,4}
SELECT ARRAY[1,2,3] @> ARRAY[2];  -- true (contains)
SELECT ARRAY[2] <@ ARRAY[1,2,3];  -- true (is contained by)
SELECT ARRAY[1,2,3] && ARRAY[2,3,4];  -- true (overlaps)
```

### JSON/JSONB

```sql
-- Create JSON
SELECT '{"name": "John", "age": 30}'::JSON;
SELECT '{"name": "John", "age": 30}'::JSONB;  -- Binary, faster

-- Access JSON
SELECT data->>'name' AS name FROM users;  -- Text
SELECT data->'address'->>'city' AS city FROM users;  -- Nested

-- JSON functions
SELECT JSON_EXTRACT_PATH_TEXT('{"a":{"b":"value"}}', 'a', 'b');  -- 'value'

-- JSONB operators
SELECT '{"a": 1}'::JSONB @> '{"a": 1}'::JSONB;  -- true (contains)
SELECT '{"a": 1, "b": 2}'::JSONB - 'b';  -- {"a": 1}
SELECT '{"a": 1}'::JSONB || '{"b": 2}'::JSONB;  -- {"a":1,"b":2}

-- Query JSONB columns
SELECT * FROM products WHERE data->>'category' = 'electronics';
SELECT * FROM products WHERE data @> '{"category": "electronics"}';

-- Create index on JSONB
CREATE INDEX idx_products_data ON products USING GIN (data);
```

### Generate Series

```sql
-- Number series
SELECT * FROM generate_series(1, 10);
SELECT * FROM generate_series(0, 100, 10);  -- Step by 10

-- Date series
SELECT * FROM generate_series(
    '2025-01-01'::DATE,
    '2025-12-31'::DATE,
    INTERVAL '1 day'
);

-- Timestamp series
SELECT * FROM generate_series(
    '2025-01-01 00:00:00'::TIMESTAMP,
    '2025-01-01 23:00:00'::TIMESTAMP,
    INTERVAL '1 hour'
);
```

---

## Performance & Optimization

### EXPLAIN

```sql
-- Show query plan
EXPLAIN SELECT * FROM employees WHERE salary > 70000;

-- Show actual execution
EXPLAIN ANALYZE SELECT * FROM employees WHERE salary > 70000;

-- More details
EXPLAIN (ANALYZE, BUFFERS, VERBOSE)
SELECT * FROM employees WHERE salary > 70000;
```

### Query Optimization Tips

```sql
-- Use indexes effectively
CREATE INDEX idx_employees_salary ON employees(salary);

-- Use LIMIT when possible
SELECT * FROM employees ORDER BY salary DESC LIMIT 10;

-- Avoid SELECT *
SELECT employee_id, first_name, last_name FROM employees;  -- Better

-- Use EXISTS instead of IN for correlated queries
SELECT * FROM departments d
WHERE EXISTS (SELECT 1 FROM employees e WHERE e.department_id = d.department_id);

-- Use appropriate JOIN type
-- INNER JOIN when you need matching rows only
-- LEFT JOIN when you need all from left table

-- Avoid functions on indexed columns in WHERE
-- Bad:  WHERE UPPER(last_name) = 'SMITH'
-- Good: WHERE last_name = 'Smith' (or create expression index)

-- Use UNION ALL instead of UNION when duplicates are OK
SELECT first_name FROM employees
UNION ALL  -- Faster
SELECT first_name FROM contractors;

-- Partition large tables
CREATE TABLE measurements_y2025_m01 PARTITION OF measurements
FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
```

### Statistics & Maintenance

```sql
-- Update statistics
ANALYZE employees;

-- Vacuum (reclaim storage)
VACUUM employees;

-- Full vacuum (locks table)
VACUUM FULL employees;

-- Vacuum and analyze together
VACUUM ANALYZE employees;

-- Check table bloat
SELECT schemaname, tablename,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

---

## Common Interview Patterns

### Find Nth Highest/Lowest

```sql
-- 2nd highest salary
SELECT DISTINCT salary
FROM employees
ORDER BY salary DESC
LIMIT 1 OFFSET 1;

-- Using window function
SELECT DISTINCT salary
FROM (
    SELECT salary, DENSE_RANK() OVER (ORDER BY salary DESC) AS rank
    FROM employees
) AS ranked
WHERE rank = 2;

-- Nth highest per department
SELECT *
FROM (
    SELECT
        department_id,
        employee_id,
        salary,
        DENSE_RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS rank
    FROM employees
) AS ranked
WHERE rank = 2;
```

### Find Duplicates

```sql
-- Find duplicate emails
SELECT email, COUNT(*) AS count
FROM employees
GROUP BY email
HAVING COUNT(*) > 1;

-- Get all rows with duplicate emails
SELECT e.*
FROM employees e
JOIN (
    SELECT email
    FROM employees
    GROUP BY email
    HAVING COUNT(*) > 1
) AS dupes ON e.email = dupes.email;

-- Using window function
SELECT *
FROM (
    SELECT *, COUNT(*) OVER (PARTITION BY email) AS email_count
    FROM employees
) AS counted
WHERE email_count > 1;
```

### Delete Duplicates (Keep One)

```sql
-- Keep row with lowest ID
DELETE FROM employees
WHERE employee_id NOT IN (
    SELECT MIN(employee_id)
    FROM employees
    GROUP BY email
);

-- Using window function
DELETE FROM employees
WHERE employee_id IN (
    SELECT employee_id
    FROM (
        SELECT employee_id, ROW_NUMBER() OVER (PARTITION BY email ORDER BY employee_id) AS rn
        FROM employees
    ) AS numbered
    WHERE rn > 1
);
```

### Running Total

```sql
-- Simple running total
SELECT
    hire_date,
    salary,
    SUM(salary) OVER (ORDER BY hire_date) AS running_total
FROM employees;

-- Running total per department
SELECT
    department_id,
    hire_date,
    salary,
    SUM(salary) OVER (PARTITION BY department_id ORDER BY hire_date) AS dept_running_total
FROM employees;
```

### Gap and Islands

```sql
-- Find gaps in sequence
WITH numbered AS (
    SELECT
        employee_id,
        employee_id - ROW_NUMBER() OVER (ORDER BY employee_id) AS grp
    FROM employees
)
SELECT
    MIN(employee_id) AS island_start,
    MAX(employee_id) AS island_end,
    MAX(employee_id) - MIN(employee_id) + 1 AS island_size
FROM numbered
GROUP BY grp
ORDER BY island_start;
```

### Pivot Table

```sql
-- Simulate pivot (rows to columns)
SELECT
    EXTRACT(YEAR FROM hire_date) AS year,
    COUNT(*) FILTER (WHERE department_id = 1) AS dept_1,
    COUNT(*) FILTER (WHERE department_id = 2) AS dept_2,
    COUNT(*) FILTER (WHERE department_id = 3) AS dept_3
FROM employees
GROUP BY EXTRACT(YEAR FROM hire_date);

-- Using CASE
SELECT
    EXTRACT(YEAR FROM hire_date) AS year,
    SUM(CASE WHEN department_id = 1 THEN 1 ELSE 0 END) AS dept_1,
    SUM(CASE WHEN department_id = 2 THEN 1 ELSE 0 END) AS dept_2,
    SUM(CASE WHEN department_id = 3 THEN 1 ELSE 0 END) AS dept_3
FROM employees
GROUP BY EXTRACT(YEAR FROM hire_date);
```

### Unpivot Table

```sql
-- Using UNION ALL
SELECT department_id, 'Q1' AS quarter, q1_sales AS sales FROM dept_sales
UNION ALL
SELECT department_id, 'Q2', q2_sales FROM dept_sales
UNION ALL
SELECT department_id, 'Q3', q3_sales FROM dept_sales
UNION ALL
SELECT department_id, 'Q4', q4_sales FROM dept_sales;

-- Using CROSS JOIN with VALUES
SELECT
    department_id,
    quarter,
    CASE quarter
        WHEN 'Q1' THEN q1_sales
        WHEN 'Q2' THEN q2_sales
        WHEN 'Q3' THEN q3_sales
        WHEN 'Q4' THEN q4_sales
    END AS sales
FROM dept_sales
CROSS JOIN (VALUES ('Q1'), ('Q2'), ('Q3'), ('Q4')) AS q(quarter);
```

### Hierarchical Queries (Employee Management)

```sql
-- Get all employees under a manager (recursive)
WITH RECURSIVE subordinates AS (
    -- Anchor: direct reports
    SELECT employee_id, first_name, manager_id, 1 AS level
    FROM employees
    WHERE manager_id = 5  -- Starting manager

    UNION ALL

    -- Recursive: reports of reports
    SELECT e.employee_id, e.first_name, e.manager_id, s.level + 1
    FROM employees e
    JOIN subordinates s ON e.manager_id = s.employee_id
)
SELECT * FROM subordinates;

-- Get management chain for an employee
WITH RECURSIVE management_chain AS (
    -- Anchor: the employee
    SELECT employee_id, first_name, manager_id, 0 AS level
    FROM employees
    WHERE employee_id = 100

    UNION ALL

    -- Recursive: their managers
    SELECT e.employee_id, e.first_name, e.manager_id, mc.level + 1
    FROM employees e
    JOIN management_chain mc ON e.employee_id = mc.manager_id
)
SELECT * FROM management_chain ORDER BY level DESC;
```

### Moving Average

```sql
-- 3-row moving average
SELECT
    hire_date,
    salary,
    AVG(salary) OVER (
        ORDER BY hire_date
        ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
    ) AS moving_avg_3
FROM employees;

-- 7-day moving average
SELECT
    date,
    sales,
    AVG(sales) OVER (
        ORDER BY date
        RANGE BETWEEN INTERVAL '3 days' PRECEDING AND INTERVAL '3 days' FOLLOWING
    ) AS moving_avg_7d
FROM daily_sales;
```

### Median Calculation

```sql
-- Using PERCENTILE_CONT
SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) AS median_salary
FROM employees;

-- Per department
SELECT
    department_id,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) AS median_salary
FROM employees
GROUP BY department_id;

-- Using window function
SELECT DISTINCT
    department_id,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary)
        OVER (PARTITION BY department_id) AS median_salary
FROM employees;
```

### Cumulative Distribution

```sql
-- Percentile rank
SELECT
    employee_id,
    salary,
    PERCENT_RANK() OVER (ORDER BY salary) AS percentile_rank,
    CUME_DIST() OVER (ORDER BY salary) AS cumulative_dist
FROM employees;

-- What percentage earn less than X?
SELECT
    salary,
    CUME_DIST() OVER (ORDER BY salary) * 100 AS percent_less_than
FROM employees
WHERE salary = 75000;
```

### Self-Join (Comparing Rows)

```sql
-- Find employees hired on same day
SELECT
    e1.first_name AS emp1,
    e2.first_name AS emp2,
    e1.hire_date
FROM employees e1
JOIN employees e2
    ON e1.hire_date = e2.hire_date
    AND e1.employee_id < e2.employee_id;  -- Avoid duplicates

-- Find employees in same department
SELECT
    e1.first_name AS emp1,
    e2.first_name AS colleague,
    e1.department_id
FROM employees e1
JOIN employees e2
    ON e1.department_id = e2.department_id
    AND e1.employee_id <> e2.employee_id;
```

---

## Quick Reference Tables

### JOIN Types Summary

| JOIN Type | Returns |
|-----------|---------|
| INNER JOIN | Only matching rows from both tables |
| LEFT JOIN | All from left + matches from right (NULL if no match) |
| RIGHT JOIN | All from right + matches from left (NULL if no match) |
| FULL OUTER JOIN | All rows from both (NULL where no match) |
| CROSS JOIN | All combinations (Cartesian product) |

### Aggregate Functions Summary

| Function | Description |
|----------|-------------|
| COUNT(*) | Count all rows |
| COUNT(col) | Count non-NULL values |
| SUM(col) | Sum values |
| AVG(col) | Average |
| MIN(col) | Minimum |
| MAX(col) | Maximum |
| STRING_AGG(col, delim) | Concatenate strings |
| ARRAY_AGG(col) | Aggregate to array |

### Window Functions Summary

| Function | Description |
|----------|-------------|
| ROW_NUMBER() | Sequential numbering |
| RANK() | Ranking with gaps |
| DENSE_RANK() | Ranking without gaps |
| NTILE(n) | Divide into n buckets |
| LEAD(col, n) | Next row value |
| LAG(col, n) | Previous row value |
| FIRST_VALUE(col) | First in partition |
| LAST_VALUE(col) | Last in partition |

### Execution Order

```
1. FROM (including JOINs)
2. WHERE
3. GROUP BY
4. HAVING
5. SELECT
6. DISTINCT
7. ORDER BY
8. LIMIT / OFFSET
```

### Key Performance Tips

1. ✅ Create indexes on columns in WHERE, JOIN, ORDER BY
2. ✅ Use EXPLAIN ANALYZE to check query plans
3. ✅ Avoid SELECT * - specify needed columns
4. ✅ Use LIMIT when testing or paginating
5. ✅ Use EXISTS instead of IN for correlated subqueries
6. ✅ Use appropriate data types (don't use VARCHAR for numbers)
7. ✅ Regularly VACUUM and ANALYZE tables
8. ❌ Don't use functions on indexed columns in WHERE
9. ❌ Don't retrieve more data than needed
10. ❌ Don't create too many indexes (slow INSERT/UPDATE/DELETE)

---

## Interview Preparation Checklist

### Must Know
- ✅ SELECT with WHERE, ORDER BY, LIMIT
- ✅ All JOIN types (especially INNER and LEFT)
- ✅ GROUP BY with aggregate functions
- ✅ HAVING clause
- ✅ Basic subqueries
- ✅ Window functions (ROW_NUMBER, RANK, DENSE_RANK)
- ✅ CASE expressions
- ✅ NULL handling
- ✅ Primary and foreign keys
- ✅ Indexes basics

### Should Know
- ✅ CTEs (WITH clause)
- ✅ Recursive CTEs
- ✅ Set operations (UNION, INTERSECT, EXCEPT)
- ✅ Correlated subqueries
- ✅ EXISTS / NOT EXISTS
- ✅ String and date functions
- ✅ Constraints (CHECK, UNIQUE, etc.)
- ✅ Transactions
- ✅ Views

### Nice to Know
- ✅ Advanced window functions (LEAD, LAG, FIRST_VALUE)
- ✅ FILTER clause
- ✅ LATERAL joins
- ✅ DISTINCT ON
- ✅ JSON/JSONB operations
- ✅ Array operations
- ✅ Performance tuning basics
- ✅ Materialized views

---

**This cheatsheet covers comprehensive SQL/PostgreSQL concepts for technical interviews. Practice with real databases and always understand WHY a query works, not just HOW to write it!**
