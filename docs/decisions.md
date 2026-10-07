-- 10/06/2026 --
Initial framework build and configuration.
 - Embed ticket information (short description, time created, time closed, ticket type) using bge-small-en-v1.5 AND MiniLM-L6.
 - Compare advantage and consistency between models.
 - Train logistic regression on vectors embedded.
 - Run cosine similarity or kNN over vectors.
 - Export.

Approval received from Supervisor prior to project start.

Data files are cleaned of EID and name data as accurately as possible, then extra precaution taken by adding training data into .gitignore.

Sensitive data contained and not exposed to public servers.

Data used: Ticket type, short description, time open.

