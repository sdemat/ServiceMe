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

-- 10/07/2026 --

After messing around with Jupyter testing and model training, decided that there are realistically only three high priority entries:
Similar tickets, time estimates, and priority.

The model will still guess the category and record source, but these aren't exactly required and will have more lenient restrictions on data input.

-- 10/08/2026 --

More Jupyter testing and config changes yielded interesting results
1. Use a cutoff for similar tickets instead of a static number when estimating time.
2. Attempt to suggest descriptions to normalize data for future entries.
3. Decided that it is more okay to give a time estimate on an inquiry than say "at the window" for an incident, so prioritize incident catches.

80% incident accuracy is acceptable; realistically, 90% is impossible with unregulated short descriptions. It is hard to correctly guess based on irregular short description naming without a large language model.

RESULT  TIER  TEST                                         DETAIL
------------------------------------------------------------------------------------------------------------------------
PASS    MUST  end to end: what the technician sees         weighted cost 136 vs 161 for the cheaper lazy strategy (need 10% lower); 18 incidents shown AT THE WINDOW, 173 inquiries given a time \
PASS    MUST  entry: priority                              '4 - Low' is 99%: use a fixed default. Uncommon-value recall 50% (need 50% for a model to help) \
PASS    MUST  similar tickets (minilm)                     top-3 hit 75% vs random 34% (need +10%), non-dominant tickets \
PASS    MUST  similar tickets (tf-idf)                     top-3 hit 78% vs random 34% (need +10%), non-dominant tickets \
PASS    MUST  stage 1: inquiry vs incident                 incidents caught 87% (need 85%), informedness 50% (need 30%; 0% = constant guess), AUC 0.83 \
PASS    MUST  time estimate: incidents (minilm)            all >= 0.40 similar (87% close, rest nearest 10): within 2x 42% vs 34% best baseline (need +5%), within-1-bucket 64% vs 52% (need +10%) \
PASS    MUST  time estimate: incidents (tf-idf)            all >= 0.40 similar (61% close, rest nearest 10): within 2x 46% vs 34% best baseline (need +5%), within-1-bucket 69% vs 52% (need +10%) \
PASS    NICE  entry: category                              top-1 89% (default 75%), top-3 96% (need 80%) \
PASS    NICE  entry: record_source                         top-1 80% (default 50%), top-3 94% (need 80%) \
FAIL    NICE  suggested description                        46% covered (need 50%), same suggestion for near-identical tickets 94% (need 80%), meaning unchanged 94% (need 90%), PII hits 0 \
PASS    NICE  time estimate: best case (minilm)            10% best case: ticket took at least that long 82% (need 75%) \
PASS    NICE  time estimate: best case (tf-idf)            10% best case: ticket took at least that long 78% (need 75%) 