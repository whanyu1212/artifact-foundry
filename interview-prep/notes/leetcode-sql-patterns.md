# LeetCode SQL Patterns - Interview Prep

**Quick reference for LeetCode SQL 50 & Advanced SQL 50 questions organized by pattern. PostgreSQL syntax with BigQuery differences highlighted.**

---

## ⚠️ PostgreSQL vs BigQuery - Key Differences

### Critical Syntax Differences

| Feature | PostgreSQL | BigQuery | Watch Out! |
|---------|-----------|----------|------------|
| **String Concatenation** | `\|\|` or `CONCAT()` | `\|\|` or `CONCAT()` | ✅ Same |
| **LIMIT** | `LIMIT n OFFSET m` | `LIMIT n OFFSET m` | ✅ Same |
| **ILIKE (case-insensitive)** | `ILIKE` | Not supported | ❌ Use `LOWER()` in BigQuery |
| **Date Functions** | `EXTRACT(YEAR FROM date)` | `EXTRACT(YEAR FROM date)` | ✅ Same |
| **Date Arithmetic** | `date + INTERVAL '1 day'` | `DATE_ADD(date, INTERVAL 1 DAY)` | ❌ Different! |
| **String to Date** | `TO_DATE('2025-01-01', 'YYYY-MM-DD')` | `PARSE_DATE('%Y-%m-%d', '2025-01-01')` | ❌ Different format codes! |
| **Date Truncation** | `DATE_TRUNC('month', date)` | `DATE_TRUNC(date, MONTH)` | ❌ Reversed params! |
| **REGEXP** | `~ 'pattern'` or `REGEXP_REPLACE` | `REGEXP_CONTAINS` | ❌ Different syntax |
| **MEDIAN** | `PERCENTILE_CONT(0.5)` | `PERCENTILE_CONT(col, 0.5)` | ❌ Different syntax |
| **UPSERT** | `ON CONFLICT DO UPDATE` | `MERGE` | ❌ Completely different |
| **Auto-increment** | `SERIAL` | Not supported | ❌ BigQuery is append-only |
| **Window Frame Default** | `RANGE UNBOUNDED PRECEDING` | `RANGE UNBOUNDED PRECEDING` | ✅ Same |
| **String Split** | `SPLIT_PART(str, ',', 1)` | `SPLIT(str, ',')[OFFSET(0)]` | ❌ 0-indexed in BQ! |
| **Arrays** | 1-indexed `arr[1]` | 0-indexed `arr[OFFSET(0)]` | ❌ Critical difference! |
| **Type Casting** | `::INTEGER` or `CAST(x AS INTEGER)` | `CAST(x AS INT64)` | ❌ Different type names |
| **Boolean Values** | `TRUE`, `FALSE`, `'t'`, `'f'` | `TRUE`, `FALSE` only | ❌ No string bools in BQ |

### Date Format Codes Comparison

| Component | PostgreSQL | BigQuery |
|-----------|-----------|----------|
| Year (4-digit) | `YYYY` | `%Y` |
| Month (2-digit) | `MM` | `%m` |
| Day (2-digit) | `DD` | `%d` |
| Hour (24h) | `HH24` | `%H` |
| Minute | `MI` | `%M` |
| Second | `SS` | `%S` |

---

## Table of Contents

1. [Select & Filtering](#1-select--filtering)
2. [Joins](#2-joins)
3. [Aggregation & GROUP BY](#3-aggregation--group-by)
4. [Subqueries](#4-subqueries)
5. [Window Functions](#5-window-functions)
6. [String Manipulation](#6-string-manipulation)
7. [Date/Time Operations](#7-datetime-operations)
8. [Self Joins](#8-self-joins)
9. [CTEs & Recursive Queries](#9-ctes--recursive-queries)
10. [Advanced Patterns](#10-advanced-patterns)

---

## 1. Select & Filtering

### Pattern: Basic SELECT with WHERE
**Representative Questions**: Recyclable and Low Fat Products (1757), Find Customer Referee (584)

```sql
-- Pattern: Filter with multiple conditions
SELECT product_id
FROM Products
WHERE low_fats = 'Y' AND recyclable = 'Y';

-- Pattern: NULL handling
SELECT name
FROM Customer
WHERE referee_id IS NULL OR referee_id <> 2;

-- 🚨 PostgreSQL vs BigQuery: Boolean columns
-- PostgreSQL: Can use 't'/'f', TRUE/FALSE, or boolean type
-- BigQuery: Must use TRUE/FALSE only
```

**Key Takeaways**:
- Always use `IS NULL` / `IS NOT NULL` for NULL checks, never `= NULL`
- `<>` and `!=` are equivalent (prefer `<>` for SQL standard)
- NULL is not equal to anything, including itself

---

## 2. Joins

### Pattern: INNER JOIN
**Representative Questions**: Replace Employee ID (1378), Students and Examinations (1280)

```sql
-- Pattern: Simple INNER JOIN
SELECT e.unique_id, em.name
FROM Employees em
LEFT JOIN EmployeeUNI e ON em.id = e.id;

-- Pattern: Multiple joins with counting
SELECT
    s.student_id,
    s.student_name,
    sub.subject_name,
    COALESCE(COUNT(e.student_id), 0) AS attended_exams
FROM Students s
CROSS JOIN Subjects sub
LEFT JOIN Examinations e
    ON s.student_id = e.student_id
    AND sub.subject_name = e.subject_name
GROUP BY s.student_id, s.student_name, sub.subject_name
ORDER BY s.student_id, sub.subject_name;
```

**Key Takeaways**:
- `LEFT JOIN` returns all from left table (NULLs if no match)
- `CROSS JOIN` creates all combinations (Cartesian product)
- Always specify join condition to avoid accidental Cartesian products
- Use `COALESCE()` to handle NULL counts

### Pattern: Finding Non-Matching Records
**Representative Questions**: Customers Who Never Order (183), Article Views (1148)

```sql
-- Pattern 1: LEFT JOIN with NULL check
SELECT c.name AS Customers
FROM Customers c
LEFT JOIN Orders o ON c.id = o.customerId
WHERE o.id IS NULL;

-- Pattern 2: NOT IN (watch for NULLs!)
SELECT name AS Customers
FROM Customers
WHERE id NOT IN (SELECT DISTINCT customerId FROM Orders WHERE customerId IS NOT NULL);

-- Pattern 3: NOT EXISTS (preferred for correlated checks)
SELECT name AS Customers
FROM Customers c
WHERE NOT EXISTS (
    SELECT 1 FROM Orders o WHERE o.customerId = c.id
);
```

**Key Takeaways**:
- `NOT IN` fails if subquery contains NULL - always filter NULLs or use `NOT EXISTS`
- `NOT EXISTS` is generally more efficient for correlated queries
- `LEFT JOIN + IS NULL` is intuitive and performant

---

## 3. Aggregation & GROUP BY

### Pattern: Basic Aggregation
**Representative Questions**: Invalid Tweets (1683), Customer Placing Largest Orders (586)

```sql
-- Pattern: COUNT with filtering
SELECT tweet_id
FROM Tweets
WHERE LENGTH(content) > 15;

-- Pattern: Find mode (most frequent value)
SELECT customer_number
FROM Orders
GROUP BY customer_number
ORDER BY COUNT(*) DESC
LIMIT 1;

-- 🚨 PostgreSQL vs BigQuery: LENGTH
-- Both support LENGTH() - same syntax
```

### Pattern: Aggregation with Conditions
**Representative Questions**: Classes More Than 5 Students (596), Sales Person (607)

```sql
-- Pattern: GROUP BY with HAVING
SELECT class
FROM Courses
GROUP BY class
HAVING COUNT(student) >= 5;

-- Pattern: Aggregation with exclusion
SELECT s.name
FROM SalesPerson s
WHERE s.sales_id NOT IN (
    SELECT o.sales_id
    FROM Orders o
    JOIN Company c ON o.com_id = c.com_id
    WHERE c.name = 'RED'
);
```

**Key Takeaways**:
- `WHERE` filters before grouping, `HAVING` filters after
- `HAVING` can use aggregate functions, `WHERE` cannot
- Use `COUNT(*)` for all rows, `COUNT(column)` for non-NULL only

### Pattern: Conditional Aggregation
**Representative Questions**: User Activity (1141), Game Play Analysis (511)

```sql
-- Pattern: COUNT with CASE
SELECT
    activity_date AS day,
    COUNT(DISTINCT user_id) AS active_users
FROM Activity
WHERE activity_date BETWEEN '2019-06-28' AND '2019-07-27'
GROUP BY activity_date;

-- Pattern: MIN/MAX with GROUP BY
SELECT
    player_id,
    MIN(event_date) AS first_login
FROM Activity
GROUP BY player_id;
```

---

## 4. Subqueries

### Pattern: Scalar Subquery (Single Value)
**Representative Questions**: Second Highest Salary (176), Department Highest Salary (184)

```sql
-- Pattern: Nth highest with OFFSET
SELECT
    COALESCE(
        (SELECT DISTINCT salary
         FROM Employee
         ORDER BY salary DESC
         LIMIT 1 OFFSET 1),
        NULL
    ) AS SecondHighestSalary;

-- Pattern: Subquery in WHERE
SELECT d.name AS Department, e.name AS Employee, e.salary AS Salary
FROM Employee e
JOIN Department d ON e.departmentId = d.id
WHERE (e.departmentId, e.salary) IN (
    SELECT departmentId, MAX(salary)
    FROM Employee
    GROUP BY departmentId
);
```

**Key Takeaways**:
- Use `COALESCE()` or outer SELECT to return NULL when no result
- `LIMIT 1 OFFSET n-1` gets the Nth row
- Tuple comparison `(col1, col2) IN (...)` for multiple columns

### Pattern: Correlated Subquery
**Representative Questions**: Employees Earning More Than Managers (181), Duplicate Emails (182)

```sql
-- Pattern: Correlated subquery (references outer query)
SELECT e1.name AS Employee
FROM Employee e1
WHERE e1.salary > (
    SELECT e2.salary
    FROM Employee e2
    WHERE e2.id = e1.managerId
);

-- Pattern: Finding duplicates
SELECT email AS Email
FROM Person
GROUP BY email
HAVING COUNT(*) > 1;
```

---

## 5. Window Functions

### Pattern: Ranking Functions
**Representative Questions**: Rank Scores (178), Department Top Three Salaries (185)

```sql
-- Pattern: DENSE_RANK for ranking without gaps
SELECT
    score,
    DENSE_RANK() OVER (ORDER BY score DESC) AS rank
FROM Scores
ORDER BY score DESC;

-- Pattern: Partition + Ranking
SELECT
    d.name AS Department,
    e.name AS Employee,
    e.salary AS Salary
FROM (
    SELECT
        *,
        DENSE_RANK() OVER (PARTITION BY departmentId ORDER BY salary DESC) AS rk
    FROM Employee
) e
JOIN Department d ON e.departmentId = d.id
WHERE e.rk <= 3;
```

**Key Takeaways**:
- `ROW_NUMBER()`: Unique sequential numbers (1, 2, 3, 4, ...)
- `RANK()`: Gaps after ties (1, 2, 2, 4, ...)
- `DENSE_RANK()`: No gaps after ties (1, 2, 2, 3, ...)
- Use `PARTITION BY` for separate rankings per group

### Pattern: LAG/LEAD for Row Comparisons
**Representative Questions**: Consecutive Numbers (180), Rising Temperature (197)

```sql
-- Pattern: Compare with previous row using LAG
SELECT DISTINCT num AS ConsecutiveNums
FROM (
    SELECT
        num,
        LAG(num, 1) OVER (ORDER BY id) AS prev1,
        LAG(num, 2) OVER (ORDER BY id) AS prev2
    FROM Logs
) t
WHERE num = prev1 AND num = prev2;

-- Pattern: Compare consecutive dates
SELECT w1.id
FROM Weather w1
JOIN Weather w2
    ON w1.recordDate = w2.recordDate + INTERVAL '1 day'
WHERE w1.temperature > w2.temperature;

-- 🚨 PostgreSQL vs BigQuery: Date arithmetic
-- PostgreSQL: w2.recordDate + INTERVAL '1 day'
-- BigQuery: DATE_ADD(w2.recordDate, INTERVAL 1 DAY)
```

**Key Takeaways**:
- `LAG(col, n)`: Value from n rows before
- `LEAD(col, n)`: Value from n rows after
- Default offset is 1 if not specified
- Third parameter is default value if NULL: `LAG(col, 1, 0)`

### Pattern: Running Totals & Moving Aggregates
**Representative Questions**: Active Businesses (1126), Human Traffic of Stadium (601)

```sql
-- Pattern: Running total
SELECT
    id,
    SUM(num) OVER (ORDER BY id) AS running_total
FROM Numbers;

-- Pattern: Moving average (3-row window)
SELECT
    id,
    AVG(num) OVER (
        ORDER BY id
        ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
    ) AS moving_avg
FROM Numbers;
```

---

## 6. String Manipulation

### Pattern: String Functions
**Representative Questions**: Fix Names (1667), Patients with Conditions (1527)

```sql
-- Pattern: Capitalize first letter
SELECT
    user_id,
    CONCAT(UPPER(LEFT(name, 1)), LOWER(SUBSTRING(name FROM 2))) AS name
FROM Users
ORDER BY user_id;

-- 🚨 PostgreSQL vs BigQuery: SUBSTRING
-- PostgreSQL: SUBSTRING(str FROM start FOR length)
-- BigQuery: SUBSTR(str, start, length)

-- Pattern: Pattern matching with LIKE
SELECT *
FROM Patients
WHERE conditions LIKE 'DIAB1%'
   OR conditions LIKE '% DIAB1%';

-- 🚨 PostgreSQL vs BigQuery: Case-insensitive search
-- PostgreSQL: ILIKE 'pattern'
-- BigQuery: Use LOWER(col) LIKE LOWER('pattern')
```

**Key Takeaways**:
- `LIKE` patterns: `%` = any chars, `_` = single char
- PostgreSQL has `ILIKE` for case-insensitive, BigQuery doesn't
- `CONCAT()` handles NULLs better than `||` in some DBs
- `LEFT(str, n)` = `SUBSTRING(str FROM 1 FOR n)`

### Pattern: String Splitting
**Representative Questions**: Group Sold Products (1484)

```sql
-- Pattern: String aggregation with ordering
SELECT
    sell_date,
    COUNT(DISTINCT product) AS num_sold,
    STRING_AGG(DISTINCT product, ',' ORDER BY product) AS products
FROM Activities
GROUP BY sell_date
ORDER BY sell_date;

-- 🚨 PostgreSQL vs BigQuery: String aggregation
-- PostgreSQL: STRING_AGG(col, delimiter ORDER BY col)
-- BigQuery: STRING_AGG(col, delimiter ORDER BY col) - same!
```

---

## 7. Date/Time Operations

### Pattern: Date Extraction & Filtering
**Representative Questions**: Monthly Transactions (1193), Immediate Food Delivery (1173)

```sql
-- Pattern: Extract year-month
SELECT
    TO_CHAR(trans_date, 'YYYY-MM') AS month,
    country,
    COUNT(*) AS trans_count,
    COUNT(*) FILTER (WHERE state = 'approved') AS approved_count,
    SUM(amount) AS trans_total_amount,
    COALESCE(SUM(amount) FILTER (WHERE state = 'approved'), 0) AS approved_total_amount
FROM Transactions
GROUP BY TO_CHAR(trans_date, 'YYYY-MM'), country;

-- 🚨 PostgreSQL vs BigQuery: Date formatting
-- PostgreSQL: TO_CHAR(date, 'YYYY-MM')
-- BigQuery: FORMAT_DATE('%Y-%m', date)

-- Pattern: First value by date
SELECT
    ROUND(
        100.0 * SUM(CASE WHEN order_date = customer_pref_delivery_date THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS immediate_percentage
FROM (
    SELECT *,
           RANK() OVER (PARTITION BY customer_id ORDER BY order_date) AS rk
    FROM Delivery
) t
WHERE rk = 1;
```

**Key Takeaways**:
- PostgreSQL: `DATE_TRUNC('month', date)` for first day of month
- BigQuery: `DATE_TRUNC(date, MONTH)` - **reversed parameter order!**
- PostgreSQL: `EXTRACT(YEAR FROM date)` = BigQuery: `EXTRACT(YEAR FROM date)` ✅
- Use `FILTER (WHERE ...)` for conditional aggregation (cleaner than CASE)

### Pattern: Date Arithmetic
**Representative Questions**: Customer Who Visited but Did Not Make Transaction (1581)

```sql
-- Pattern: Date difference
SELECT v.customer_id, COUNT(*) AS count_no_trans
FROM Visits v
LEFT JOIN Transactions t ON v.visit_id = t.visit_id
WHERE t.transaction_id IS NULL
GROUP BY v.customer_id;

-- Pattern: Date range filtering
SELECT *
FROM Orders
WHERE order_date >= CURRENT_DATE - INTERVAL '30 days';

-- 🚨 PostgreSQL vs BigQuery: Date arithmetic
-- PostgreSQL: date + INTERVAL '1 day', date - INTERVAL '30 days'
-- BigQuery: DATE_ADD(date, INTERVAL 1 DAY), DATE_SUB(date, INTERVAL 30 DAY)
```

---

## 8. Self Joins

### Pattern: Comparing Rows Within Same Table
**Representative Questions**: Delete Duplicate Emails (196), Tree Node (608)

```sql
-- Pattern: Delete duplicates, keep lowest ID
DELETE FROM Person
WHERE id NOT IN (
    SELECT MIN(id)
    FROM Person
    GROUP BY email
);

-- Alternative: Using window function
DELETE FROM Person
WHERE id IN (
    SELECT id
    FROM (
        SELECT id, ROW_NUMBER() OVER (PARTITION BY email ORDER BY id) AS rn
        FROM Person
    ) t
    WHERE rn > 1
);

-- Pattern: Self-join for classification
SELECT
    id,
    CASE
        WHEN p_id IS NULL THEN 'Root'
        WHEN id IN (SELECT DISTINCT p_id FROM Tree WHERE p_id IS NOT NULL) THEN 'Inner'
        ELSE 'Leaf'
    END AS type
FROM Tree;
```

**Key Takeaways**:
- Self-join: Join table to itself with different aliases
- Useful for finding patterns within same table
- Common for hierarchical data (employee-manager, parent-child)

---

## 9. CTEs & Recursive Queries

### Pattern: Common Table Expressions (CTEs)
**Representative Questions**: Market Analysis (1158), Exchange Seats (626)

```sql
-- Pattern: Multiple CTEs for readability
WITH first_orders AS (
    SELECT
        buyer_id,
        MIN(order_date) AS first_order_date
    FROM Orders
    GROUP BY buyer_id
),
joined_data AS (
    SELECT
        u.user_id AS buyer_id,
        u.join_date,
        COALESCE(f.first_order_date, NULL) AS first_order_date
    FROM Users u
    LEFT JOIN first_orders f ON u.user_id = f.buyer_id
)
SELECT
    buyer_id,
    join_date,
    COUNT(o.order_id) AS orders_in_2019
FROM joined_data j
LEFT JOIN Orders o
    ON j.buyer_id = o.buyer_id
    AND EXTRACT(YEAR FROM o.order_date) = 2019
GROUP BY buyer_id, join_date;
```

### Pattern: Recursive CTE
**Representative Questions**: Manager Hierarchy, Fibonacci Sequence

```sql
-- Pattern: Recursive employee hierarchy
WITH RECURSIVE employee_hierarchy AS (
    -- Base case: Top-level employees (no manager)
    SELECT
        employee_id,
        name,
        manager_id,
        1 AS level
    FROM Employees
    WHERE manager_id IS NULL

    UNION ALL

    -- Recursive case: Employees with managers
    SELECT
        e.employee_id,
        e.name,
        e.manager_id,
        eh.level + 1
    FROM Employees e
    JOIN employee_hierarchy eh ON e.manager_id = eh.employee_id
)
SELECT * FROM employee_hierarchy
ORDER BY level, employee_id;

-- Pattern: Generate number sequence
WITH RECURSIVE numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n < 100
)
SELECT * FROM numbers;

-- 🚨 PostgreSQL vs BigQuery: Recursive CTEs
-- PostgreSQL: Full recursive CTE support ✅
-- BigQuery: Limited recursive CTE support (max 500 iterations)
```

**Key Takeaways**:
- CTEs improve readability vs nested subqueries
- Can reference CTEs multiple times (computed once)
- Recursive CTEs need: base case + UNION ALL + recursive case
- Always include termination condition in recursive case!

---

## 10. Advanced Patterns

### Pattern: Pivot (Rows to Columns)
**Representative Questions**: Reformat Department Table (1179)

```sql
-- Pattern: Conditional aggregation for pivot
SELECT
    id,
    SUM(CASE WHEN month = 'Jan' THEN revenue END) AS Jan_Revenue,
    SUM(CASE WHEN month = 'Feb' THEN revenue END) AS Feb_Revenue,
    SUM(CASE WHEN month = 'Mar' THEN revenue END) AS Mar_Revenue,
    SUM(CASE WHEN month = 'Apr' THEN revenue END) AS Apr_Revenue,
    SUM(CASE WHEN month = 'May' THEN revenue END) AS May_Revenue,
    SUM(CASE WHEN month = 'Jun' THEN revenue END) AS Jun_Revenue,
    SUM(CASE WHEN month = 'Jul' THEN revenue END) AS Jul_Revenue,
    SUM(CASE WHEN month = 'Aug' THEN revenue END) AS Aug_Revenue,
    SUM(CASE WHEN month = 'Sep' THEN revenue END) AS Sep_Revenue,
    SUM(CASE WHEN month = 'Oct' THEN revenue END) AS Oct_Revenue,
    SUM(CASE WHEN month = 'Nov' THEN revenue END) AS Nov_Revenue,
    SUM(CASE WHEN month = 'Dec' THEN revenue END) AS Dec_Revenue
FROM Department
GROUP BY id;

-- Alternative: Using FILTER (PostgreSQL specific)
SELECT
    id,
    SUM(revenue) FILTER (WHERE month = 'Jan') AS Jan_Revenue,
    SUM(revenue) FILTER (WHERE month = 'Feb') AS Feb_Revenue
    -- ... etc
FROM Department
GROUP BY id;
```

### Pattern: Finding Gaps in Sequences
**Representative Questions**: Find Missing IDs, Consecutive Available Seats (603)

```sql
-- Pattern: Find consecutive available seats
SELECT DISTINCT s1.seat_id
FROM Cinema s1
JOIN Cinema s2
    ON ABS(s1.seat_id - s2.seat_id) = 1
WHERE s1.free = 1 AND s2.free = 1
ORDER BY s1.seat_id;

-- Pattern: Find gaps in sequence
WITH numbered AS (
    SELECT
        id,
        id - ROW_NUMBER() OVER (ORDER BY id) AS grp
    FROM TableName
)
SELECT
    MIN(id) AS start_id,
    MAX(id) AS end_id
FROM numbered
GROUP BY grp
ORDER BY start_id;
```

### Pattern: Median Calculation
**Representative Questions**: Median Employee Salary (569)

```sql
-- Pattern 1: PERCENTILE_CONT (preferred)
SELECT
    company_id,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) AS median_salary
FROM Employee
GROUP BY company_id;

-- 🚨 PostgreSQL vs BigQuery: PERCENTILE_CONT
-- PostgreSQL: PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col)
-- BigQuery: PERCENTILE_CONT(col, 0.5) OVER() - different syntax!

-- Pattern 2: Using ROW_NUMBER (portable)
WITH ordered AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY company_id ORDER BY salary) AS rn_asc,
        ROW_NUMBER() OVER (PARTITION BY company_id ORDER BY salary DESC) AS rn_desc
    FROM Employee
)
SELECT company_id, AVG(salary) AS median_salary
FROM ordered
WHERE rn_asc BETWEEN rn_desc - 1 AND rn_desc + 1
GROUP BY company_id;
```

### Pattern: Percentage Calculations
**Representative Questions**: Percentage of Users Attended Contest (1633)

```sql
-- Pattern: Calculate percentage with rounding
SELECT
    contest_id,
    ROUND(
        100.0 * COUNT(DISTINCT user_id) / (SELECT COUNT(*) FROM Users),
        2
    ) AS percentage
FROM Register
GROUP BY contest_id
ORDER BY percentage DESC, contest_id;

-- Key: Multiply by 100.0 (not 100) to force float division
```

### Pattern: CASE with Swap
**Representative Questions**: Swap Salary (627), Exchange Seats (626)

```sql
-- Pattern: Swap values with CASE
UPDATE Salary
SET sex = CASE
    WHEN sex = 'm' THEN 'f'
    WHEN sex = 'f' THEN 'm'
END;

-- Pattern: Swap seat IDs with CASE
SELECT
    CASE
        WHEN id % 2 = 1 AND id = (SELECT MAX(id) FROM Seat) THEN id
        WHEN id % 2 = 1 THEN id + 1
        ELSE id - 1
    END AS id,
    student
FROM Seat
ORDER BY id;
```

---

## Common Interview Pitfalls

### 1. NULL Handling
```sql
-- ❌ WRONG: NULL comparisons
SELECT * FROM table WHERE col = NULL;  -- Returns nothing!
SELECT * FROM table WHERE col != NULL; -- Returns nothing!

-- ✅ CORRECT
SELECT * FROM table WHERE col IS NULL;
SELECT * FROM table WHERE col IS NOT NULL;

-- ❌ WRONG: NOT IN with NULLs
SELECT * FROM a WHERE id NOT IN (SELECT id FROM b);
-- Fails if b.id contains NULL!

-- ✅ CORRECT
SELECT * FROM a WHERE id NOT IN (SELECT id FROM b WHERE id IS NOT NULL);
-- OR use NOT EXISTS:
SELECT * FROM a WHERE NOT EXISTS (SELECT 1 FROM b WHERE b.id = a.id);
```

### 2. GROUP BY Columns
```sql
-- ❌ WRONG: SELECT columns not in GROUP BY (PostgreSQL strict mode)
SELECT name, department, COUNT(*)
FROM employees
GROUP BY department;  -- Error! name is not aggregated or grouped

-- ✅ CORRECT
SELECT department, COUNT(*)
FROM employees
GROUP BY department;
```

### 3. Window Function vs Aggregate
```sql
-- ❌ WRONG: Using window function in WHERE
SELECT name, salary
FROM employees
WHERE ROW_NUMBER() OVER (ORDER BY salary DESC) <= 3;  -- Error!

-- ✅ CORRECT: Use subquery
SELECT name, salary
FROM (
    SELECT name, salary, ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn
    FROM employees
) t
WHERE rn <= 3;
```

### 4. Date Arithmetic
```sql
-- 🚨 PostgreSQL vs BigQuery
-- PostgreSQL:
SELECT date_col + INTERVAL '1 day' FROM table;
SELECT date_col - INTERVAL '1 month' FROM table;

-- BigQuery:
SELECT DATE_ADD(date_col, INTERVAL 1 DAY) FROM table;
SELECT DATE_SUB(date_col, INTERVAL 1 MONTH) FROM table;
```

### 5. Division by Zero
```sql
-- ❌ WRONG: Can cause division by zero
SELECT sales / employees FROM departments;

-- ✅ CORRECT: Use NULLIF or CASE
SELECT sales / NULLIF(employees, 0) FROM departments;
-- OR
SELECT CASE WHEN employees = 0 THEN NULL ELSE sales / employees END FROM departments;
```

---

## Quick Problem Categorization

### Must Practice (SQL 50 Core)

**SELECT & WHERE**: 1757, 584, 595, 1148, 1683
**JOINS**: 1378, 1280, 1068, 577, 1661, 183, 1581, 1251
**AGGREGATION**: 1795, 1407, 586, 1393, 1211
**SUBQUERIES**: 176, 1978, 626
**WINDOW FUNCTIONS**: 1321, 1341, 1398
**CASE**: 1873, 627

### Advanced SQL 50 Focus

**Window Functions Advanced**: 178, 534, 180, 1321, 601, 185
**Recursive CTE**: Hierarchical queries, number generation
**Pivot/Unpivot**: 1179, 618
**Date/Time Manipulation**: 1193, 1934, 1107, 1158
**String Operations**: 1667, 1527, 1484
**Median/Percentile**: 569, 571, 1174

---

## Practice Strategy

### Week 1: Fundamentals
- SELECT, WHERE, ORDER BY, LIMIT
- Basic JOINs (INNER, LEFT)
- GROUP BY with aggregates
- Focus: Questions 1-20 of SQL 50

### Week 2: Intermediate
- Subqueries (scalar, correlated)
- Advanced JOINs (self-join, multiple joins)
- Window functions (ROW_NUMBER, RANK)
- Focus: Questions 21-40 of SQL 50

### Week 3: Advanced
- Window functions (LAG/LEAD, frames)
- CTEs and recursive queries
- Complex aggregations
- Focus: Remaining SQL 50 + Advanced SQL 1-25

### Week 4: Expert
- Advanced window functions
- Pivot/Unpivot patterns
- Complex date/time operations
- Focus: Advanced SQL 26-50

### Before Interview
- Review PostgreSQL vs BigQuery differences
- Practice top 5 from each category
- Do 2-3 random problems cold (no notes)
- Time yourself: aim for 10-15 min per medium problem

---

## Interview Day Checklist

✅ **Syntax differences memorized** (date arithmetic, array indexing)
✅ **NULL handling** - always check for NULL in NOT IN
✅ **Window functions** - know when to use vs GROUP BY
✅ **Know execution order**: FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT
✅ **Common patterns ready**: Nth highest, duplicates, pivoting, running totals
✅ **Edge cases**: Empty tables, all NULLs, single row

**Good luck! Remember: Explain your thought process, mention edge cases, and optimize after correctness!**
