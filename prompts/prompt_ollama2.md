<|start_header_id|>system<|end_header_id|>
Follow instructions to the letter, and answer questions without making any additional assumptions.

<|start_header_id|>user<|end_header_id|>
Generate a {db_type} query to answer this question: `{user_question}`
{instructions}
DDL statements:
{table_metadata_string}

Generate a valid {db_type} query that best answers the question `{user_question}`.

Return **only** the query in a single code block (triple backticks).  
**Do not** include any explanations or text outside the code block.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```sql
-- Write only the SQL query here
