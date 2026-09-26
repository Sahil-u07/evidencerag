\# Database Systems



\## Relational Databases



A relational database stores information using relations commonly represented as tables. Rows represent records and columns represent attributes.



SQL provides operations for querying and modifying relational data. SELECT retrieves data, INSERT adds records, UPDATE modifies records, and DELETE removes records.



\## Indexing



An index is an auxiliary data structure that helps locate records without scanning every row in a table.



A B-tree index keeps keys in sorted order and supports efficient equality and range queries. Database systems can use indexes to reduce the amount of data that must be examined.



\## Transactions



A transaction is a logical unit of database work. The ACID properties describe important transaction guarantees.



Atomicity means a transaction is treated as an all-or-nothing operation. Consistency means database constraints remain satisfied. Isolation controls how concurrent transactions interact. Durability means committed changes survive failures.



\## Concurrency Control



Multiple transactions may execute concurrently. Without appropriate control, concurrent execution can produce anomalies such as dirty reads, non-repeatable reads, and lost updates.



Locking is one technique for controlling concurrent access. Shared locks can allow multiple readers, while exclusive locks provide write access to a resource.



\## Normalization



Normalization is the process of organizing relational data to reduce unnecessary redundancy and dependency problems.



First Normal Form requires atomic attribute values. Second Normal Form addresses partial dependencies on part of a composite key. Third Normal Form reduces transitive dependencies.



\## Query Optimization



A database optimizer selects an execution strategy for a SQL query. It may consider available indexes, join algorithms, estimated cardinalities, filtering conditions, and access paths.



Good statistics and appropriate indexes can significantly influence the chosen execution plan.

